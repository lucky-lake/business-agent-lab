"""Send a request to a language model through the OpenRouter API.

02_calculator_call.py and 03_assistant.py use request_model() from this file.
01_model_request.py writes the same request out in full, so you can read every part.

The API key comes from the .env file in the project root (business-agent-lab/.env).
This file never prints the key.
"""
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# The endpoint is the address of the OpenRouter API.
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# A free model that supports tool calling. To try another model, add a line such as
# OPENROUTER_MODEL=openrouter/free to the .env file. No code change is needed.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# business-agent-lab/.env is one folder above this file, wherever you run the script from.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)
MODEL = os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL


class ModelRequestError(Exception):
    """The model request did not return a reply."""


def api_key():
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ModelRequestError(f"OPENROUTER_API_KEY is missing. Add it to {ENV_FILE}")
    return key


def request_model(messages, tools=None):
    """Send the conversation (and the tool definitions) to the model.

    Returns the model's reply message as a dict. The reply has either
    "content" (text for the user) or "tool_calls" (functions the model asks us to run).
    """
    body = {"model": MODEL, "messages": messages}
    if tools:
        body["tools"] = tools
    headers = {"Authorization": f"Bearer {api_key()}"}

    for attempt in range(3):
        try:
            response = requests.post(ENDPOINT, headers=headers, json=body, timeout=60)
        except requests.RequestException as error:
            raise ModelRequestError(f"could not reach OpenRouter ({type(error).__name__})")
        if response.status_code != 429 or attempt == 2:
            break
        # 429 means "too many requests". Wait, then try again (at most three tries).
        wait = retry_after(response, default=2 ** (attempt + 1))
        print(f"(rate limited, waiting {wait} seconds)")
        time.sleep(wait)

    data = read_json(response)
    if response.status_code != 200 or "error" in data:
        raise ModelRequestError(f"HTTP {response.status_code}: {error_message(data)}")
    return data["choices"][0]["message"]


def retry_after(response, default):
    try:
        return min(float(response.headers.get("Retry-After", default)), 30)
    except ValueError:
        return default


def read_json(response):
    try:
        return response.json()
    except ValueError:
        return {"error": {"message": response.text[:200]}}


def error_message(data):
    error = data.get("error") or {}
    if isinstance(error, dict):
        return error.get("message") or "no message"
    return str(error)

