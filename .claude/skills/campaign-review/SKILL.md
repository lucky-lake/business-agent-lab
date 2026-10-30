---
name: campaign-review
description: Weekly ad campaign review that compares two weeks of campaign results (one row per campaign per week) and writes a Markdown report with CTR, conversion rate, cost per purchase and ROAS for each campaign, the change between the weeks, findings to investigate, and data problems. Use this skill whenever the user asks what changed in their ads, campaigns or marketing between weeks, wants a week-over-week or weekly campaign comparison, or asks to analyze, review or report on a campaign history file such as week-06/data/history.csv, even if they don't say "campaign review" or name the measures.
---

# Campaign review

Compare two weeks of ad campaign results and save a Markdown report. The goal is a report a person can check by hand and use to decide what to look into next. It lays out the numbers and raises questions. It does not make decisions.

## Calculation script

Run `scripts/analyze_campaigns.py` (in this skill's folder) at the start of every review, before writing anything. It covers steps 1–3 below: it picks the weeks, runs the data checks on the whole file and calculates the four measures, so the numbers in the report come from code, not from mental arithmetic.

```sh
python3 <skill folder>/scripts/analyze_campaigns.py <data.csv>                          # two latest weeks
python3 <skill folder>/scripts/analyze_campaigns.py <data.csv> --weeks 2026-09-28 2026-10-05   # weeks the user named
```

It prints Markdown with three parts: **Data checks** (every problem found, or "No problems found"), **Comparison table** and **Inputs used**, already formatted as in step 3. It only reads the CSV. If it stops with an error (for example, a week that is not in the file), tell the user what it said rather than guessing.

Copy the tables into the report and write the Findings yourself (step 4); the script gives no findings, because those take judgement. For order value, divide revenue by purchases from the Inputs table. If Python is not available, do steps 1–3 by hand as described.

## 1. Pick the weeks

The data file has one row per campaign per week. Expected columns:
`week_start, campaign, impressions, clicks, attributed_purchases, ad_spend_usd, attributed_revenue_usd`.

- If the user names two weeks, compare those.
- Otherwise compare the two latest `week_start` values in the file.

Leave every other week out of the comparison table and the findings. Earlier weeks can tempt you to talk about trends, but the report is about these two weeks only, which keeps it short and easy to check.

If a campaign appears in only one of the two weeks, keep it in the table, show the missing week as "—", and say so under Data problems.

## 2. Check the data first

Before calculating anything, check for:

- **Missing values**: empty cells in any column.
- **Negative values**: any numeric column below zero.
- **Duplicate rows**: fully repeated rows, and the same campaign appearing more than once in the same week.

Check the whole file, and say which rows you checked. Report every problem you find under Data problems. If there are none, say that the checks ran and found nothing, so the reader knows they were done.

Do not edit, clean or overwrite the data file. If a problem stops a measure from being calculated (for example, zero purchases makes cost per purchase undefined), show "n/a" in the table and explain why under Data problems.

## 3. Calculate the four measures

For each campaign in each of the two weeks:

| Measure                   | Formula                         |
|---------------------------|---------------------------------|
| Click-through rate (CTR)  | clicks ÷ impressions            |
| Conversion rate           | attributed purchases ÷ clicks   |
| Cost per purchase         | ad spend ÷ attributed purchases |
| Return on ad spend (ROAS) | attributed revenue ÷ ad spend   |

Do the arithmetic with a short script rather than by eye, so the numbers are exact. Then show:

- CTR and conversion rate as percentages with two decimals; change in percentage points (pp).
- Cost per purchase in USD with two decimals; change in USD.
- ROAS with two decimals; change as an absolute difference.
- The relative change (%) in brackets after each absolute change.

## 4. Write the report

Save a Markdown file to the path the user gives. If they don't give one, use `reports/<second week_start>.md` next to the data folder and tell the user where it went.

Use this structure:

```markdown
# Weekly campaign review: <week 1> vs <week 2>

Source: <data file>. Weeks compared: <week 1> and <week 2>. Other weeks left out.

## 1. Comparison table
One row per campaign per measure: Campaign | Measure | <week 1> | <week 2> | Change.
Below it, a table of the raw inputs used (impressions, clicks, purchases, spend, revenue
for each campaign and week), so the reader can check any number by hand.

## 2. Findings
Up to three changes worth investigating, biggest first.

## 3. Data problems
What was checked, and what was found.
```

### Findings

Pick at most three changes, the ones that moved most or matter most. For each:

- State what changed, with the numbers (for example "clicks fell from 900 to 480 while impressions rose").
- Point to which part of the funnel moved: reach (impressions), clicks, conversion, order value (revenue ÷ purchases) or spend. This helps the reader see where to look.
- Write possible causes **as questions** ("Did the ad creative change this week?"), not as statements. The data shows what happened, not why, so a question is honest about that and invites the person who knows the campaign to answer.

If a campaign barely moved, one line saying so is enough.

### What to leave out

Do not recommend budget changes: no "shift spend to", "pause", "scale up" or similar. Budget calls depend on goals and context outside this file, and the person reading the report makes them.

## 5. Tell the user

After saving, give a short summary: where the report is, whether the data checks found anything, and the one or two biggest changes. Offer one cost per purchase they can check by hand (ad spend ÷ attributed purchases).
