"""Week 05, step 2: the model requests a calculation, and Python runs it.

Run from the project root:   python week-05/02_calculator_call.py

1. The program sends a question and the calculator's tool definition.
2. The model replies with a tool call: a function name and arguments.
3. The program stops. Press Enter, and Python runs the calculator.
4. The program sends the result back, and the model writes the final answer.
"""
import json

from model_api import ModelRequestError, request_model
from tools import CALCULATOR, calculator

QUESTION = "What is 247 times 38?"

# The functions this program allows the model to request.
FUNCTIONS = {"calculator": calculator}


def describe(call):
    """calculator(operation="multiply", a=247, b=38)"""
    name = call["function"]["name"]
    try:
        arguments = read_arguments(call)
        shown = ", ".join(f"{key}={json.dumps(value)}" for key, value in arguments.items())
    except ValueError:
        shown = call["function"]["arguments"]
    return f"{name}({shown})"


def read_arguments(call):
    arguments = call["function"]["arguments"] or "{}"
    if isinstance(arguments, dict):
        return arguments
    return json.loads(arguments)


def dispatch(call):
    """Run the function that the tool call names. Return the result as text."""
    name = call["function"]["name"]
    if name not in FUNCTIONS:
        return f"error: unknown function {name}"
    try:
        arguments = read_arguments(call)
    except ValueError:
        return "error: the arguments are not valid JSON"
    try:
        result = FUNCTIONS[name](**arguments)
    except Exception as error:
        return f"error: {error}"
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def main():
    messages = [{"role": "user", "content": QUESTION}]
    print("User:", QUESTION)

    reply = request_model(messages, tools=[CALCULATOR])
    if not reply.get("tool_calls"):
        print("The model answered without a tool call:")
        print(reply.get("content"))
        return

    print("Assistant message:")
    print(json.dumps({"tool_calls": reply["tool_calls"]}, indent=2))
    for call in reply["tool_calls"]:
        print("Requested:", describe(call))
    input("Press Enter to run the function.")

    # Keep the assistant's request in the history, then add one result per call.
    messages.append({"role": "assistant", "content": reply.get("content"), "tool_calls": reply["tool_calls"]})
    for call in reply["tool_calls"]:
        result = dispatch(call)
        print("Result:", result)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    final = request_model(messages, tools=[CALCULATOR])
    print("Final answer:", final.get("content"))


if __name__ == "__main__":
    try:
        main()
    except ModelRequestError as error:
        raise SystemExit(f"Model request failed: {error}")
