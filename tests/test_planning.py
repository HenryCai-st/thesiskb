"""Plan scheduling and task matching (Milestone 15), no database."""

from datetime import date, timedelta

import pytest

from thesiskb.planning import parse_excluded, same_task, schedule


def test_excluded_days_ranges_and_iso_weeks():
    days = parse_excluded(["2026-12-24", "2026-12-30..2027-01-01", "2026-W52", ""])
    assert date(2026, 12, 24) in days and date(2027, 1, 1) in days
    assert {date(2026, 12, 21) + timedelta(days=i) for i in range(7)} <= days  # week 52
    assert date(2026, 12, 29) not in days


def test_schedule_packs_backwards_from_the_deadline():
    # 7 h/week = 1 h/day; parts of 2 h, 3 h: the last ends the day before the deadline
    slots = schedule([2, 3], date(2026, 12, 15), hours_per_week=7)
    assert [(s.start, s.end) for s in slots] == [
        (date(2026, 12, 10), date(2026, 12, 11)),
        (date(2026, 12, 12), date(2026, 12, 14)),
    ]


def test_schedule_counts_holidays_but_skips_excluded_days():
    # Dec 25/26 count like any day unless excluded
    plain = schedule([2], date(2026, 12, 27), hours_per_week=7)
    assert (plain[0].start, plain[0].end) == (date(2026, 12, 25), date(2026, 12, 26))
    skip = schedule([2], date(2026, 12, 27), 7, parse_excluded(["2026-12-24..2026-12-26"]))
    assert (skip[0].start, skip[0].end) == (date(2026, 12, 22), date(2026, 12, 23))


def test_schedule_shares_a_day_between_small_parts():
    # 14 h/week = 2 h/day: two 1 h parts fit into the same last day
    slots = schedule([1, 1], date(2026, 12, 15), hours_per_week=14)
    assert all(s.start == s.end == date(2026, 12, 14) for s in slots)


def test_schedule_rejects_zero_hours():
    with pytest.raises(ValueError):
        schedule([1], date(2026, 12, 15), hours_per_week=0)


@pytest.mark.parametrize(
    "a, b, same",
    [
        ("Submit the expose to Prof. Weber", "Expose submission to Prof. Weber", True),
        ("Submit expose to Prof. Weber", "submit the expose to Weber", True),
        ("Interview five users", "Interview five users about AI tools", True),
        ("Draft chapter 2", "Draft chapter 3", False),  # different numbers
        ("Read papers", "Read the survey", False),  # one shared word of two
    ],
)
def test_same_task(a, b, same):
    assert same_task(a, b) is same
