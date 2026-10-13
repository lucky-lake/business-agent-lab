"""Week 05, step 3: a small terminal assistant that uses several tools.

Run from the project root:   python week-05/03_assistant.py

Every tool call prints a short trace: the call, the result, and the assistant's answer.
The conversation is kept only while the program runs.
"""
import json
from datetime import date

from model_api import ModelRequestError, request_model
from tools import (
    CALCULATOR,
    CONVERT_CURRENCY,
    READ_SCHEDULE,
    SAVE_NOTE,
    SEARCH_WIKIPEDIA,
    calculator,
    convert_currency,
    read_schedule,
    save_note,
    search_wikipedia,
)

# The most tool rounds for one user message. The assistant stops after this many.
MAX_ROUNDS = 5

# The tool definitions sent to the model.
TOOLS = [CALCULATOR, READ_SCHEDULE, SEARCH_WIKIPEDIA, SAVE_NOTE, CONVERT_CURRENCY]

# The functions this program allows the model to request.
FUNCTIONS = {
    "calculator": calculator,
    "read_schedule": read_schedule,
    "search_wikipedia": search_wikipedia,
    "save_note": save_note,
    "convert_currency": convert_currency,
}

SYSTEM_PROMPT = (
    f"You are a small terminal assistant for one student. Today is {date.today():%A, %Y-%m-%d}. "
    "Use a tool when the request needs information or an action that a tool provides. "
    "Use the calculator for arithmetic. "
    "If a request is missing information that you need, ask one short question instead of guessing. "
    "If a tool returns an error, say that the action failed. Do not invent a result. "
    "Keep answers short. Write plain text, without Markdown."
)


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Assistant ready. Type exit to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit"}:
            break
        if text:
            messages.append({"role": "user", "content": text})
            answer(messages)


def answer(messages):
    """Request the model until it answers without a tool call, or until MAX_ROUNDS."""
    start = len(messages) - 1
    for _ in range(MAX_ROUNDS):
        try:
            reply = request_model(messages, TOOLS)
        except ModelRequestError as error:
            trace("error", f"model request failed: {error}")
            del messages[start:]  # forget this message, so the history stays valid
            return
        messages.append({"role": "assistant", "content": reply.get("content"), "tool_calls": reply.get("tool_calls")})
        if not reply.get("tool_calls"):
            messages[-1].pop("tool_calls")
            trace("assistant", reply.get("content") or "(no text)")
            return
        for call in reply["tool_calls"]:
            trace("tool call", describe(call))
            result = dispatch(call)
            trace("tool result", shorten(result))
            messages.append(tool_result(call["id"], result))
    trace("stopped", f"no final answer after {MAX_ROUNDS} tool rounds")


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


def tool_result(call_id, result):
    """The message that returns a result to the model. The id matches the call."""
    return {"role": "tool", "tool_call_id": call_id, "content": result}


def read_arguments(call):
    arguments = call["function"]["arguments"] or "{}"
    if isinstance(arguments, dict):
        return arguments
    return json.loads(arguments)


def describe(call):
    """calculator(operation="divide", a=10, b=0)"""
    name = call["function"]["name"]
    try:
        arguments = read_arguments(call)
        shown = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in arguments.items())
    except ValueError:
        shown = call["function"]["arguments"]
    return f"{name}({shown})"


def shorten(text, limit=160):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def trace(label, text):
    print(f"{label:<13}{text}")


if __name__ == "__main__":
    main()
