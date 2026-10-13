"""Week 05, step 1: send one request to a language model and read the response.

Run from the project root:   python week-05/01_model_request.py

Find three things in this file: the endpoint, the messages, and the response.
"""
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load the API key from business-agent-lab/.env (one folder above this script).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# The model. model_api.py uses the same default.
MODEL = os.getenv("OPENROUTER_MODEL") or "nvidia/nemotron-3-super-120b-a12b:free"

# 1. The endpoint: the address of the OpenRouter API.
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def main():
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY is missing. Add it to business-agent-lab/.env")

    # 2. The request body: which model to use, and the messages.
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }

    # The authentication header carries the API key. Never print it.
    headers = {"Authorization": f"Bearer {API_KEY}"}

    print("Endpoint:", ENDPOINT)
    print("Request body:")
    print(json.dumps(body, indent=2))

    response = requests.post(ENDPOINT, headers=headers, json=body, timeout=60)
    print("HTTP status:", response.status_code)
    data = response.json()
    if response.status_code != 200 or "error" in data:
        print("The request failed:", data.get("error"))
        raise SystemExit(1)

    # 3. The response: the model's reply is in choices[0]["message"].
    message = data["choices"][0]["message"]
    print("Response message:")
    print(json.dumps({"role": message["role"], "content": message["content"]}, indent=2, ensure_ascii=False))
    print("Model used:", data.get("model"))


if __name__ == "__main__":
    main()
