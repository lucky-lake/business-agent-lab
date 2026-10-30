# Week 06: Skills

This folder goes inside your `business-agent-lab` folder:

```text
business-agent-lab/
  .claude/skills/        project skills (campaign-review is saved here)
  week-06/               this folder
    analysis-method.md   the weekly campaign review method
    prompts.md           the prompts for the exercise
    data/history.csv     twelve weeks of campaign results
    reports/             reports that Claude Code writes
```

The data is synthetic. It describes the ad campaigns of a small online store.

## Setup

1. Extract `week-06.zip` into `business-agent-lab`.
2. Open `business-agent-lab` in VS Code.
3. Start Claude Code in the terminal, from the `business-agent-lab` folder:

   ```sh
   claude
   ```

   Check that Claude Code names `business-agent-lab` as its folder. The script in step 5 needs Python 3; `python3 --version` should print a version.

You do not need an API key this week.

## Exercise

Follow the prompts in `prompts.md`, in order.

1. Check one cost per purchase by hand: ad spend ÷ attributed purchases.
2. Run the first review with the method file. It compares the weeks of 28 September and 5 October. The data has no missing, negative, or duplicate rows; the review checks anyway and says so.
3. Create the `campaign-review` skill with skill-creator. Use the skill-creator name that `/skills` shows on your computer. If `/campaign-review` does not appear afterwards, run `/reload-skills`.
4. Read the saved `SKILL.md`. Find the cost per purchase formula and the report sections.
5. Add the calculation script.
6. Start a new Claude Code session (`/exit`, then `claude`) and run the fresh-session test. It compares the two latest weeks.
7. Start another new session and run the last prompt. It names neither the skill nor the method. Look for `Skill(campaign-review)` in the output.

Check each report against your hand calculation. A skill does not make the result correct.
