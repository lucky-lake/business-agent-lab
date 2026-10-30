#!/usr/bin/env python3
"""Data checks and the four campaign measures for a two-week comparison.

Usage:
    python3 analyze_campaigns.py <history.csv> [--weeks WEEK1 WEEK2]

Without --weeks, the two latest week_start values are compared.
Prints Markdown: data checks, comparison table and raw inputs table.
Reads the CSV only; it never writes to it.
"""

import argparse
import csv
import sys
from collections import Counter

COLUMNS = [
    "week_start",
    "campaign",
    "impressions",
    "clicks",
    "attributed_purchases",
    "ad_spend_usd",
    "attributed_revenue_usd",
]
NUMERIC = COLUMNS[2:]
MINUS = "−"
# measure: (numerator column, denominator column)
FORMULA = {
    "CTR": ("clicks", "impressions"),
    "Conversion rate": ("attributed_purchases", "clicks"),
    "Cost per purchase": ("ad_spend_usd", "attributed_purchases"),
    "ROAS": ("attributed_revenue_usd", "ad_spend_usd"),
}


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        # Line 1 is the header, so data rows start at line 2.
        rows = [(n, row) for n, row in enumerate(reader, start=2)]
    return header, rows


def check(header, rows):
    """Return a list of problem strings for the whole file."""
    problems = []
    missing_cols = [c for c in COLUMNS if c not in header]
    if missing_cols:
        problems.append(f"Missing columns: {', '.join(missing_cols)}")

    for n, row in rows:
        for col in COLUMNS:
            if col not in header:
                continue
            value = (row.get(col) or "").strip()
            if value == "":
                problems.append(f"Missing value: line {n}, column {col}")
                continue
            if col in NUMERIC:
                try:
                    number = float(value)
                except ValueError:
                    problems.append(f"Not a number: line {n}, column {col} = {value!r}")
                    continue
                if number < 0:
                    problems.append(f"Negative value: line {n}, column {col} = {value}")

    full = Counter(tuple((row.get(c) or "").strip() for c in COLUMNS) for _, row in rows)
    for values, count in full.items():
        if count > 1:
            lines = [n for n, row in rows if tuple((row.get(c) or "").strip() for c in COLUMNS) == values]
            problems.append(f"Duplicate row ({count} copies): lines {', '.join(map(str, lines))}")

    keys = Counter((row.get("week_start"), row.get("campaign")) for _, row in rows)
    for (week, campaign), count in keys.items():
        if count > 1:
            lines = [n for n, row in rows if (row.get("week_start"), row.get("campaign")) == (week, campaign)]
            problems.append(
                f"Campaign {campaign} appears {count} times in week {week}: lines {', '.join(map(str, lines))}"
            )
    return problems


def to_number(value):
    try:
        return float((value or "").strip())
    except ValueError:
        return None


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def measures(row):
    return {
        measure: ratio(to_number(row.get(num)), to_number(row.get(den)))
        for measure, (num, den) in FORMULA.items()
    }


def why_na(row, measure):
    reasons = []
    for col in FORMULA[measure]:
        if to_number(row.get(col)) is None:
            reasons.append(f"{col} is missing or not a number")
    den = to_number(row.get(FORMULA[measure][1]))
    if den == 0:
        reasons.append(f"{FORMULA[measure][1]} is 0")
    return "; ".join(reasons)


def signed(text, value):
    if value > 0:
        return "+" + text
    if value < 0:
        return MINUS + text
    return text


def fmt_value(measure, value):
    if value is None:
        return "n/a"
    if measure in ("CTR", "Conversion rate"):
        return f"{value * 100:.2f}%"
    if measure == "Cost per purchase":
        return f"${value:.2f}"
    return f"{value:.2f}"


def fmt_change(measure, old, new):
    if old is None or new is None:
        return "n/a"
    diff = new - old
    if measure in ("CTR", "Conversion rate"):
        absolute = signed(f"{abs(diff) * 100:.2f} pp", round(diff * 100, 2))
    elif measure == "Cost per purchase":
        absolute = signed(f"${abs(diff):.2f}", round(diff, 2))
    else:
        absolute = signed(f"{abs(diff):.2f}", round(diff, 2))
    if old == 0:
        relative = "n/a"
    else:
        rel = (new / old - 1) * 100
        relative = signed(f"{abs(rel):.1f}%", round(rel, 1))
    return f"{absolute} ({relative})"


def fmt_raw(value):
    number = to_number(value)
    if number is None:
        return (value or "").strip() or "—"
    return f"{number:,.0f}" if number == int(number) else f"{number:,.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("--weeks", nargs=2, metavar=("WEEK1", "WEEK2"))
    args = parser.parse_args()

    try:
        header, rows = load(args.csv_path)
    except OSError as e:
        sys.exit(f"Cannot read {args.csv_path}: {e}")
    absent = [c for c in ("week_start", "campaign") if c not in header]
    if absent:
        sys.exit(f"Column(s) not in file: {', '.join(absent)}. Columns found: {', '.join(header)}")

    weeks_in_file = sorted({(row.get("week_start") or "").strip() for _, row in rows} - {""})
    if args.weeks:
        weeks = sorted(args.weeks)
        if weeks[0] == weeks[1]:
            sys.exit("--weeks needs two different weeks.")
        unknown = [w for w in weeks if w not in weeks_in_file]
        if unknown:
            sys.exit(f"Week(s) not in file: {', '.join(unknown)}. Weeks in file: {', '.join(weeks_in_file)}")
    else:
        if len(weeks_in_file) < 2:
            sys.exit("The file has fewer than two weeks.")
        weeks = weeks_in_file[-2:]
    w1, w2 = weeks

    problems = check(header, rows)

    # One row per (campaign, week); a duplicated pair is ambiguous, so it gets no measures.
    by_key = {}
    dup_keys = set()
    rows_in_weeks = []
    for _, row in rows:
        key = ((row.get("campaign") or "").strip(), (row.get("week_start") or "").strip())
        if key[1] not in weeks:
            continue
        rows_in_weeks.append((key, row))
        if key in by_key:
            dup_keys.add(key)
        by_key[key] = row
    for key in dup_keys:
        del by_key[key]

    campaigns = sorted({c for c, w in by_key} | {c for c, w in dup_keys})
    for campaign in campaigns:
        for week in weeks:
            if (campaign, week) not in by_key and (campaign, week) not in dup_keys:
                problems.append(f"Campaign {campaign} has no row in week {week}")
    for campaign, week in sorted(dup_keys):
        problems.append(f"Measures for {campaign} in week {week} not calculated: more than one row")
    for (campaign, week), row in sorted(by_key.items()):
        for measure, value in measures(row).items():
            if value is None:
                problems.append(f"{measure} for {campaign} in week {week} is n/a: {why_na(row, measure)}")

    print(f"# Campaign analysis: {w1} vs {w2}\n")
    print(f"Source: {args.csv_path}. Rows checked: {len(rows)} (whole file, "
          f"{weeks_in_file[0] if weeks_in_file else '?'} to {weeks_in_file[-1] if weeks_in_file else '?'}).\n")

    print("## Data checks\n")
    print("Checked: missing values, negative values, non-numeric values, duplicate rows, "
          "repeated campaign in the same week, campaigns missing from one of the two weeks.\n")
    if problems:
        for p in problems:
            print(f"- {p}")
    else:
        print("- No problems found.")
    print()

    print("## Comparison table\n")
    print(f"| Campaign | Measure | {w1} | {w2} | Change |")
    print("|---|---|---:|---:|---|")
    for campaign in campaigns:
        m1 = measures(by_key[(campaign, w1)]) if (campaign, w1) in by_key else None
        m2 = measures(by_key[(campaign, w2)]) if (campaign, w2) in by_key else None
        blank1 = "n/a" if (campaign, w1) in dup_keys else "—"
        blank2 = "n/a" if (campaign, w2) in dup_keys else "—"
        for measure in ("CTR", "Conversion rate", "Cost per purchase", "ROAS"):
            v1 = fmt_value(measure, m1[measure]) if m1 else blank1
            v2 = fmt_value(measure, m2[measure]) if m2 else blank2
            if m1 and m2:
                change = fmt_change(measure, m1[measure], m2[measure])
            else:
                change = "n/a" if "n/a" in (blank1, blank2) and "—" not in (v1, v2) else "—"
            print(f"| {campaign} | {measure} | {v1} | {v2} | {change} |")
    print()

    print("## Inputs used\n")
    print("| Campaign | Week | Impressions | Clicks | Purchases | Spend (USD) | Revenue (USD) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for campaign in campaigns:
        for week in weeks:
            matching = [row for key, row in rows_in_weeks if key == (campaign, week)]
            if not matching:
                print(f"| {campaign} | {week} | — | — | — | — | — |")
            for row in matching:
                cells = " | ".join(fmt_raw(row.get(col)) for col in NUMERIC)
                print(f"| {campaign} | {week} | {cells} |")


if __name__ == "__main__":
    main()
