"""Tools for the Week 05 assistant.

Each tool has two parts:
  1. a tool definition (a dict) that describes the tool to the model, and
  2. a Python function that does the work when the program runs it.

The model only reads the definitions. The program runs the functions.
"""
import html
import json
import re
from datetime import date as Date
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
SCHEDULE_FILE = HERE / "data" / "schedule.json"
NOTES_DIR = HERE / "notes"


# Calculator ---------------------------------------------------------------

CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Do one arithmetic operation on two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["operation", "a", "b"],
        },
    },
}


def calculator(operation, a, b):
    if not all(isinstance(x, (int, float)) for x in (a, b)):
        raise ValueError("a and b must be numbers")
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("division by zero")
        result = a / b
    else:
        raise ValueError(f"unknown operation: {operation}")
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return result


# Schedule reading ---------------------------------------------------------

READ_SCHEDULE = {
    "type": "function",
    "function": {
        "name": "read_schedule",
        "description": "Read the user's schedule for one date. Returns the events on that date.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "The date in YYYY-MM-DD format."},
            },
            "required": ["date"],
        },
    },
}


def read_schedule(date):
    try:
        Date.fromisoformat(date)
    except ValueError:
        raise ValueError("date must be YYYY-MM-DD")
    events = json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    found = [event for event in events if event["date"] == date]
    return found or f"no events on {date}"


# Wikipedia search ---------------------------------------------------------

SEARCH_WIKIPEDIA = {
    "type": "function",
    "function": {
        "name": "search_wikipedia",
        "description": (
            "Search English Wikipedia. Returns up to three pages with a title, URL, and short excerpt. "
            "Searches Wikipedia only, not the whole web."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search words."},
            },
            "required": ["query"],
        },
    },
}


def search_wikipedia(query):
    response = requests.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": 3},
        headers={"User-Agent": "business-agent-lab-week-05 (course exercise)"},
        timeout=10,
    )
    response.raise_for_status()
    pages = response.json().get("pages", [])
    results = [
        {
            "title": page["title"],
            "url": "https://en.wikipedia.org/wiki/" + page["key"],
            "excerpt": html.unescape(re.sub(r"<[^>]+>", "", page.get("excerpt") or "")),
        }
        for page in pages
    ]
    return results or f"no Wikipedia pages found for {query}"


# Note saving --------------------------------------------------------------

SAVE_NOTE = {
    "type": "function",
    "function": {
        "name": "save_note",
        "description": (
            "Save a short note as a Markdown file in the notes folder. "
            "Never replaces an existing file. Returns the path of the saved file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "A short file name, such as practice-note."},
                "text": {"type": "string", "description": "The text of the note."},
            },
            "required": ["filename", "text"],
        },
    },
}


def save_note(filename, text):
    # Keep only letters, digits, and hyphens, so the file always lands inside notes/.
    stem = re.sub(r"[^a-z0-9-]+", "-", Path(filename).stem.lower()).strip("-")[:40] or "note"
    NOTES_DIR.mkdir(exist_ok=True)
    path = NOTES_DIR / f"{stem}.md"
    number = 2
    while path.exists():
        path = NOTES_DIR / f"{stem}-{number}.md"
        number += 1
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return f"saved notes/{path.name}"


# Currency conversion (exercise) -------------------------------------------
#
# Add two things here:
#   1. a function           convert_currency(amount, from_currency, to_currency)
#   2. a tool definition    CONVERT_CURRENCY, written like the definitions above
# Then register both in 03_assistant.py, in TOOLS and in FUNCTIONS.
#
# Rate service: Frankfurter (no API key needed). Example request and response:
#   GET https://api.frankfurter.dev/v2/rate/USD/KRW
#   {"date": "2026-10-05", "base": "USD", "quote": "KRW", "rate": 1347.79}
#
# Return the rate, its date, and the converted amount (amount × rate).
# The rate is a dated reference rate, not the amount a bank pays after fees.

CONVERT_CURRENCY = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": (
            "Convert an amount of money from one currency to another with the latest reference rate. "
            "Returns the rate, the date of the rate, and the converted amount. "
            "The rate does not include bank fees."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "The amount to convert."},
                "from_currency": {"type": "string", "description": "The three-letter currency code to convert from, such as USD."},
                "to_currency": {"type": "string", "description": "The three-letter currency code to convert to, such as KRW."},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}


def convert_currency(amount, from_currency, to_currency):
    if not isinstance(amount, (int, float)):
        raise ValueError("amount must be a number")
    codes = []
    for code in (from_currency, to_currency):
        code = str(code).strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError(f"not a three-letter currency code: {code}")
        codes.append(code)
    base, quote = codes
    response = requests.get(
        f"https://api.frankfurter.dev/v2/rate/{base}/{quote}",
        headers={"User-Agent": "business-agent-lab-week-05 (course exercise)"},
        timeout=10,
    )
    if response.status_code == 422:
        raise ValueError(response.json().get("message", "unknown currency"))
    response.raise_for_status()
    data = response.json()
    return {
        "amount": amount,
        "from_currency": base,
        "to_currency": quote,
        "rate": data["rate"],
        "rate_date": data["date"],
        "converted_amount": round(amount * data["rate"], 2),
    }
