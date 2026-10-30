# Week 06 prompts

Copy each prompt into Claude Code at the lab root (`business-agent-lab`).

## 1. First review (method file)

```text
Read week-06/analysis-method.md.
Apply the method to week-06/data/history.csv for the weeks starting 2026-09-28 and 2026-10-05.
Save the report to week-06/reports/2026-10-05.md.
```

## 2. Skill creation

Use the skill-creator name that `/skills` shows on your computer. When skill-creator asks questions, answer briefly, skip the test runs, and let it save the file.

```text
/anthropic-skills:skill-creator Create a skill named campaign-review.
Use the method in week-06/analysis-method.md.
Save it in .claude/skills/campaign-review/ in this project.
```

## 3. Calculation script

```text
Write a Python script, scripts/analyze_campaigns.py, in .claude/skills/campaign-review/.
It should do the data checks and the four measures in SKILL.md.
Update SKILL.md to say when and how to run it. Leave the rest of SKILL.md unchanged.
```

## 4. Fresh-session test

Start a new Claude Code session at the lab root first (`/exit`, then `claude`). A new session starts with no memory of the old chat, so only the saved skill can guide it.

```text
/campaign-review Analyze the two latest weeks in week-06/data/history.csv.
Save the report to week-06/reports/2026-10-19.md.
```

## 5. Automatic selection

Start a new Claude Code session at the lab root first (`/exit`, then `claude`). This prompt does not name the skill or the method. Look for `Skill(campaign-review)` in the output.

```text
What changed in our ads over the two latest weeks in week-06/data/history.csv?
Save the report to week-06/reports/2026-10-19-auto.md.
```
