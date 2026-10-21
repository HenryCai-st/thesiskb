"""Resolve due dates as written ("Nov", "Jan 2027", "05.2027", "next Friday") to calendar dates.

Done in code, not by the LLM, so it is testable and the same input always gives the same date.
Month-only dates ("Nov", "March") take the first such month on/after an anchor date: the
earliest explicit date among the source's tasks (a schedule note starting "10.10.2026"), else
the source's own date (recording/publication date, else ingestion date). A due date that names
only a month is set to the last day of that month.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

MONTHS = {
    name: i
    for i in range(1, 13)
    for name in {
        calendar.month_name[i].lower(),
        calendar.month_abbr[i].lower(),
        # German month names, for German meetings/notes
        [
            "januar",
            "februar",
            "märz",
            "april",
            "mai",
            "juni",
            "juli",
            "august",
            "september",
            "oktober",
            "november",
            "dezember",
        ][i - 1],
        ["jan", "feb", "mär", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "dez"][i - 1],
    }
}
MONTHS["sept"] = 9
MONTHS["maerz"] = 3
WEEKDAYS = {
    **{calendar.day_name[i].lower(): i for i in range(7)},
    **dict(
        zip(
            ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"],
            range(7),
            strict=True,
        )
    ),
}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))


def end_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _year(y: str) -> int:
    y_int = int(y)
    return y_int + 2000 if y_int < 100 else y_int


def parse_explicit(raw: str) -> tuple[date, bool] | None:
    """A date that carries its year. Returns (date, month_only)."""
    s = raw.strip().lower()
    if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s):  # 2026-10-10
        return date(int(m[1]), int(m[2]), int(m[3])), False
    if m := re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b", s):  # 10.10.2026
        return date(_year(m[3]), int(m[2]), int(m[1])), False
    if m := re.search(r"\b(\d{1,2})[./](\d{4})\b", s):  # 05.2027
        return end_of_month(int(m[2]), int(m[1])), True
    if m := re.search(rf"\b(\d{{1,2}})\.?\s+({_MONTH_RE})\.?\s+(\d{{4}})\b", s):  # 3 March 2027
        return date(int(m[3]), MONTHS[m[2]], int(m[1])), False
    if m := re.search(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", s):  # March 3, 2027
        return date(int(m[3]), MONTHS[m[1]], int(m[2])), False
    if m := re.search(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", s):  # Jan 2027
        return end_of_month(int(m[2]), MONTHS[m[1]]), True
    return None


def resolve_due(raw: str | None, anchor: date) -> date | None:
    """Calendar date for a due date as written, relative to `anchor` when it has no year."""
    if not raw or not raw.strip():
        return None
    if found := parse_explicit(raw):
        return found[0]
    s = raw.strip().lower()
    if m := re.search(rf"\b(\d{{1,2}})\.?\s+({_MONTH_RE})\b|\b({_MONTH_RE})\.?\s+(\d{{1,2}})\b", s):
        day = int(m[1] or m[4])
        month = MONTHS[m[2] or m[3]]
        year = anchor.year if (month, day) >= (anchor.month, anchor.day) else anchor.year + 1
        return date(year, month, day)
    if m := re.search(rf"^(?:end of |ende )?({_MONTH_RE})\.?$", s):  # "Nov", "March"
        month = MONTHS[m[1]]
        year = anchor.year if month >= anchor.month else anchor.year + 1
        return end_of_month(year, month)
    if s in ("today", "heute"):
        return anchor
    if s in ("tomorrow", "morgen"):
        return anchor + timedelta(days=1)
    if m := re.search(r"\bin (\d+) (day|days|week|weeks)\b", s):
        n = int(m[1]) * (7 if m[2].startswith("week") else 1)
        return anchor + timedelta(days=n)
    if s in ("next week", "nächste woche"):
        return anchor + timedelta(days=7 - anchor.weekday())  # Monday of next week
    # "Friday", "next Friday", "by Friday", "bis Freitag": the first such day after the anchor
    if (m := re.search(r"(\w+)$", s)) and m[1] in WEEKDAYS:
        days = (WEEKDAYS[m[1]] - anchor.weekday()) % 7 or 7
        return anchor + timedelta(days=days)
    return None


def anchor_for(raw_dues: list[str | None], fallback: date) -> date:
    """Earliest explicit full date among a source's tasks, else the source's own date."""
    explicit = [d for r in raw_dues if r and (p := parse_explicit(r)) and not p[1] for d in [p[0]]]
    return min(explicit) if explicit else fallback
