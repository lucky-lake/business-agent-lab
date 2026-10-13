# Week 05: Tools and Function Calling

This folder goes inside your `business-agent-lab` folder:

```text
business-agent-lab/
  .gitignore
  .env                  your API key (never commit this file)
  week-05/              this folder
    01_model_request.py
    02_calculator_call.py
    03_assistant.py
    tools.py
    model_api.py        sends requests to the model API
    data/schedule.json  a fictional schedule
    notes/              notes that save_note writes
```

## Setup

Run these commands in the terminal, from the `business-agent-lab` folder.

1. Put your OpenRouter API key in `business-agent-lab/.env`, the project root, not this folder. The file has one line, as in `env.example`:

   ```text
   OPENROUTER_API_KEY=your_key_here
   ```

2. Tell Git to ignore the key file and the Python environment. Add these two lines to the end of `business-agent-lab/.gitignore`:

   ```text
   /.env
   /.venv/
   ```

   Then check. Git prints `.env` when it ignores the file. Do this before your next commit.

   ```sh
   git check-ignore .env
   ```

3. Create a Python environment and install the two packages.

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r week-05/requirements.txt
   ```

   On Ubuntu (WSL), if the first command fails, run `sudo apt install python3-venv` and try again. Run `source .venv/bin/activate` again each time you open a new terminal.

## Run

```sh
python week-05/01_model_request.py     # one request to the model
python week-05/02_calculator_call.py   # the model requests a calculation
python week-05/03_assistant.py         # the assistant; type exit to quit
```

## Exercise

Add currency conversion to the assistant.

1. Run `03_assistant.py` before you change anything.
2. In `tools.py`, add the function `convert_currency(amount, from_currency, to_currency)` and its tool definition `CONVERT_CURRENCY`. The comment at the end of `tools.py` shows the rate service.
3. In `03_assistant.py`, add `CONVERT_CURRENCY` to `TOOLS` and `convert_currency` to `FUNCTIONS`.
4. Ask the assistant to convert an amount, and read the trace: the tool call, the tool result, and the answer. Check the answer against amount × rate.

You may use Claude Code for the exercise. You explain what the code does.

### If you finish early

Add a small tool of your own. It is optional and not graded. A new tool needs the same three parts as currency conversion:

1. the function, in `tools.py`
2. its tool definition, in `tools.py`, with a `"name"` that matches the function name
3. the registration, in `TOOLS` and `FUNCTIONS` in `03_assistant.py`

Then ask the assistant a question that needs your tool, and check the trace for the tool call and the tool result.

## If something fails

* `OPENROUTER_API_KEY is missing`: the `.env` file is not in `business-agent-lab`, or the line has a typo.
* `HTTP 401`: the key is wrong or was deleted. Create a new key on OpenRouter.
* `HTTP 429`: too many requests. Wait a minute and try again. Free models allow a limited number of requests per day.
* A different model: add `OPENROUTER_MODEL=openrouter/free` to `.env`.
