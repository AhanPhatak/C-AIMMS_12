"""Resolve relative time expressions in a memory's text against its own timestamp.

A conversation turn stamped "1:56 pm on 8 May, 2023" that says "I went to a
support group yesterday" carries its date only implicitly: to answer "When did
she go?" something has to do the arithmetic. This module does it once, at
ingestion, so the stored memory reads

    I went to a support group yesterday (7 May, 2023)

The original words are kept; the resolved date is appended in parentheses.
Nothing here knows about any benchmark's questions or answers -- the only input
is each memory's own timestamp, and resolved dates are written in the SAME
style that timestamp uses (day-first with comma, day-first, month-first, or
ISO), so a dataset's memories stay in that dataset's own date format.

Deliberately rule-based and English-only (as is the rest of the pipeline):
deterministic, no LLM calls, and it cannot invent a date that the timestamp
does not imply. Expressions it cannot resolve unambiguously (e.g. "recently",
"last winter", "on the 5th of May") are left untouched.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
_MON_RE = "|".join(MONTHS)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WD_RE = "|".join(WEEKDAYS)
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_NUM_RE = r"(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"

# Season -> last month of the season (northern hemisphere). Winter spans a year
# boundary, so "last winter" is ambiguous and deliberately not resolved.
_SEASON_END = {"spring": 5, "summer": 8, "fall": 11, "autumn": 11}


# ---------------------------------------------------------------------------
# Timestamp parsing: recover the date AND the style it was written in
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DateStyle:
    kind: str          # "day_first" | "month_first" | "iso"
    comma: bool = False


_DAY_FIRST = re.compile(rf"\b(\d{{1,2}})\s+({_MON_RE}),?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_FIRST = re.compile(rf"\b({_MON_RE})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE)
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")   # not \b: "2023-05-08T10:00"


def _month_index(name: str) -> int:
    return [m.lower() for m in MONTHS].index(name.lower()) + 1


def parse_timestamp(ts: Optional[str]) -> Optional[Tuple[dt.date, DateStyle]]:
    """Date and writing style of a timestamp such as '1:56 pm on 8 May, 2023',
    'May 8, 2023' or '2023-05-08'. None if no calendar date can be found."""
    if not ts:
        return None
    try:
        m = _DAY_FIRST.search(ts)
        if m:
            d = dt.date(int(m.group(3)), _month_index(m.group(2)), int(m.group(1)))
            return d, DateStyle("day_first", comma="," in m.group(0))
        m = _MONTH_FIRST.search(ts)
        if m:
            d = dt.date(int(m.group(3)), _month_index(m.group(1)), int(m.group(2)))
            return d, DateStyle("month_first", comma="," in m.group(0))
        m = _ISO.search(ts)
        if m:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))), DateStyle("iso")
    except ValueError:  # e.g. "31 February"
        return None
    return None


def format_day(d: dt.date, style: DateStyle) -> str:
    if style.kind == "iso":
        return d.isoformat()
    month = MONTHS[d.month - 1]
    sep = "," if style.comma else ""
    if style.kind == "month_first":
        return f"{month} {d.day}{sep} {d.year}"
    return f"{d.day} {month}{sep} {d.year}"


def format_month(d: dt.date, style: DateStyle) -> str:
    if style.kind == "iso":
        return f"{d.year:04d}-{d.month:02d}"
    return f"{MONTHS[d.month - 1]} {d.year}"


def date_only(ts: Optional[str]) -> Optional[str]:
    """The timestamp's calendar date alone, in its own style: '1:56 pm on
    8 May, 2023' -> '8 May, 2023'. Unparseable timestamps are returned as-is."""
    parsed = parse_timestamp(ts)
    return format_day(*parsed) if parsed else ts


# ---------------------------------------------------------------------------
# Tense: decides "on Friday" / "the 17th" -> the one before or the one after
# ---------------------------------------------------------------------------

_FUTURE_RE = re.compile(
    r"\b(will|won't|shall|gonna|going to|plan(?:s|ned|ning)? to|planning|hope to|hoping to|"
    r"can't wait|looking forward|upcoming|about to|scheduled)\b|'ll\b",
    re.IGNORECASE,
)
def _sentence_at(text: str, pos: int) -> str:
    start = max(text.rfind(c, 0, pos) for c in ".!?") + 1
    ends = [i for i in (text.find(c, pos) for c in ".!?") if i != -1]
    return text[start:(min(ends) + 1) if ends else len(text)]


def _is_future(text: str, pos: int) -> bool:
    return bool(_FUTURE_RE.search(_sentence_at(text, pos)))


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

Resolver = Callable[[re.Match, dt.date, DateStyle, str], Optional[str]]


def _n(tok: str) -> int:
    return int(tok) if tok.isdigit() else _NUM[tok.lower()]


def _add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.month - 1 + n, 12)
    return dt.date(d.year + y, m + 1, 1)


def _day(offset: int) -> Resolver:
    return lambda m, d, s, t: format_day(d + dt.timedelta(offset), s)


def _weekday_rel(m: re.Match, d: dt.date, s: DateStyle, text: str, *, direction: Optional[str] = None) -> str:
    wd = m.group("wd").capitalize()
    if direction is None:
        direction = "after" if _is_future(text, m.start()) else "before"
    return f"the {wd} {direction} {format_day(d, s)}"


def _nth_of_month(m: re.Match, d: dt.date, s: DateStyle, text: str) -> Optional[str]:
    n = int(m.group("n"))
    if not 1 <= n <= 31:
        return None
    future = _is_future(text, m.start())
    if future:
        month_start = d.replace(day=1) if n >= d.day else _add_months(d, 1)
    else:
        month_start = d.replace(day=1) if n <= d.day else _add_months(d, -1)
    try:
        return format_day(month_start.replace(day=n), s)
    except ValueError:      # the 31st of a 30-day month: not a real date
        return None


def _season(m: re.Match, d: dt.date, s: DateStyle, text: str) -> str:
    which, season = m.group("which").lower(), m.group("season").lower()
    end = _SEASON_END[season]
    if which == "last":
        year = d.year if d.month > end else d.year - 1
    elif which == "this":
        year = d.year
    else:  # next
        year = d.year if d.month < end - 2 else d.year + 1
    return f"the {season} of {year}"


# Most specific first; each character span is resolved at most once.
_RULES: List[Tuple[str, Resolver]] = [
    (r"\bthe day before yesterday\b", _day(-2)),
    (r"\bthe day after tomorrow\b", _day(2)),
    (r"\byesterday\b", _day(-1)),
    (r"\blast night\b", _day(-1)),
    (r"\b(?:today|tonight|this (?:morning|afternoon|evening))\b", _day(0)),
    (r"\btomorrow\b", _day(1)),
    (rf"\b{_NUM_RE} days? ago\b", lambda m, d, s, t: format_day(d - dt.timedelta(_n(m.group(1))), s)),
    (r"\b(?:a few|few|a couple of|couple of|several) days ago\b|\bthe other day\b",
     lambda m, d, s, t: f"a few days before {format_day(d, s)}"),
    (rf"\b{_NUM_RE} weekends? ago\b", lambda m, d, s, t: f"{m.group(1)} weekends before {format_day(d, s)}"),
    (rf"\b{_NUM_RE} weeks? ago\b", lambda m, d, s, t: f"{m.group(1)} weeks before {format_day(d, s)}"),
    (rf"\b{_NUM_RE} months? ago\b", lambda m, d, s, t: format_month(_add_months(d, -_n(m.group(1))), s)),
    (rf"\b{_NUM_RE} years? ago\b", lambda m, d, s, t: str(d.year - _n(m.group(1)))),
    (r"\b(?:this past|past|last) weekend\b", lambda m, d, s, t: f"the weekend before {format_day(d, s)}"),
    (r"\bover the weekend\b",
     lambda m, d, s, t: f"the weekend {'after' if _is_future(t, m.start()) else 'before'} {format_day(d, s)}"),
    (r"\bthis weekend\b", lambda m, d, s, t: f"the weekend of {format_day(d, s)}"),
    (r"\bnext weekend\b", lambda m, d, s, t: f"the weekend after {format_day(d, s)}"),
    (r"\b(?:this past|past|last) week\b", lambda m, d, s, t: f"the week before {format_day(d, s)}"),
    (r"\bthis week\b", lambda m, d, s, t: f"the week of {format_day(d, s)}"),
    (r"\bnext week\b", lambda m, d, s, t: f"the week after {format_day(d, s)}"),
    (r"\blast month\b", lambda m, d, s, t: format_month(_add_months(d, -1), s)),
    (r"\bthis month\b", lambda m, d, s, t: format_month(d, s)),
    (r"\bnext month\b", lambda m, d, s, t: format_month(_add_months(d, 1), s)),
    (r"\blast year\b", lambda m, d, s, t: str(d.year - 1)),
    (r"\bthis year\b", lambda m, d, s, t: str(d.year)),
    (r"\bnext year\b", lambda m, d, s, t: str(d.year + 1)),
    (r"\b(?P<which>last|this|next) (?P<season>spring|summer|fall|autumn)\b", _season),
    (rf"\blast (?P<wd>{_WD_RE})\b", lambda m, d, s, t: _weekday_rel(m, d, s, t, direction="before")),
    (rf"\bnext (?P<wd>{_WD_RE})\b", lambda m, d, s, t: _weekday_rel(m, d, s, t, direction="after")),
    # "on/this Friday" (singular only -- \b stops "on Fridays", which is habitual):
    # before or after the timestamp depending on the sentence's tense.
    (rf"\b(?:on|this) (?P<wd>{_WD_RE})\b", _weekday_rel),
    # "on the 17th" / "the 11th" -- but not "the 11th of May", "11th May",
    # "the 5th grade", "our 10th anniversary"...
    (rf"\bthe (?P<n>\d{{1,2}})(?:st|nd|rd|th)\b"
     rf"(?!\s+(?:of\b|{_MON_RE}\b|grade|place|time|century|anniversary|birthday|floor|round|edition|year|day|week|month))",
     _nth_of_month),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), fn) for p, fn in _RULES]


def resolve_relative_time(text: str, timestamp: Optional[str]) -> Tuple[str, int]:
    """Append the resolved date after every relative time expression in
    ``text``, computed from ``timestamp``. Returns (new_text, n_resolved).

    Idempotent: an expression already followed by a parenthesised resolution
    is left alone, so re-running over migrated text changes nothing.
    """
    parsed = parse_timestamp(timestamp)
    if not text or parsed is None:
        return text, 0
    d, style = parsed
    taken: List[Tuple[int, int, str]] = []
    for pattern, fn in _COMPILED:
        for m in pattern.finditer(text):
            if any(not (m.end() <= a or m.start() >= b) for a, b, _ in taken):
                continue
            if text[m.end():m.end() + 2] == " (":
                continue   # already resolved
            resolved = fn(m, d, style, text)
            if resolved:
                taken.append((m.start(), m.end(), resolved))
    if not taken:
        return text, 0
    out, pos = [], 0
    for a, b, res in sorted(taken):
        out.append(text[pos:b])
        out.append(f" ({res})")
        pos = b
    out.append(text[pos:])
    return "".join(out), len(taken)
