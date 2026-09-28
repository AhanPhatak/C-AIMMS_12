"""Unit tests for iterret.time_resolution (relative-date resolution at ingestion).

Pure Python, no GPU. Runnable either with pytest or directly:
`python3 -m iterret.tests.test_time_resolution`.
"""
from __future__ import annotations

import sys

from iterret.time_resolution import date_only, parse_timestamp, resolve_relative_time

TS = "1:56 pm on 8 May, 2023"          # a Monday


def _r(text, ts=TS):
    return resolve_relative_time(text, ts)[0]


def test_parse_keeps_dataset_style():
    assert date_only(TS) == "8 May, 2023"
    assert date_only("May 8, 2023") == "May 8, 2023"
    assert date_only("8 May 2023") == "8 May 2023"
    assert date_only("2023-05-08T10:00") == "2023-05-08"
    assert date_only("Session 3") == "Session 3"          # unparseable: untouched
    assert parse_timestamp("31 February, 2023") is None


def test_day_level():
    assert _r("I went yesterday.") == "I went yesterday (7 May, 2023)."
    assert _r("Three days ago I left.") == "Three days ago (5 May, 2023) I left."
    assert _r("See you tomorrow!") == "See you tomorrow (9 May, 2023)!"
    assert _r("the day before yesterday") == "the day before yesterday (6 May, 2023)"


def test_output_follows_timestamp_style():
    assert _r("yesterday", "May 8, 2023") == "yesterday (May 7, 2023)"
    assert _r("yesterday", "2023-05-08") == "yesterday (2023-05-07)"
    assert _r("last month", "2023-05-08") == "last month (2023-04)"


def test_week_month_year():
    assert _r("last week") == "last week (the week before 8 May, 2023)"
    assert _r("this past weekend") == "this past weekend (the weekend before 8 May, 2023)"
    assert _r("next month") == "next month (June 2023)"
    assert _r("last month", "3 January, 2024") == "last month (December 2023)"
    assert _r("five years ago") == "five years ago (2018)"


def test_weekdays_follow_tense():
    assert _r("last Friday") == "last Friday (the Friday before 8 May, 2023)"
    assert _r("next Saturday") == "next Saturday (the Saturday after 8 May, 2023)"
    assert _r("I hosted a class on Friday.") == "I hosted a class on Friday (the Friday before 8 May, 2023)."
    assert _r("We will play on Saturday.") == "We will play on Saturday (the Saturday after 8 May, 2023)."
    assert _r("I go running on Fridays.") == "I go running on Fridays."    # habitual: untouched


def test_nth_of_month():
    assert _r("It dropped on the 5th.") == "It dropped on the 5th (5 May, 2023)."
    assert _r("It dropped on the 20th.") == "It dropped on the 20th (20 April, 2023)."
    assert _r("The show will open on the 20th.") == "The show will open on the 20th (20 May, 2023)."
    assert _r("on the 5th of June") == "on the 5th of June"
    assert _r("my 10th anniversary") == "my 10th anniversary"
    assert _r("the 31st", "3 May, 2023") == "the 31st"   # 31 April does not exist


def test_seasons():
    assert _r("last summer", "11 August, 2023") == "last summer (the summer of 2022)"
    assert _r("last summer", "2 October, 2023") == "last summer (the summer of 2023)"
    assert _r("last winter") == "last winter"


def test_unresolvable_and_idempotent():
    assert resolve_relative_time("I recently moved.", TS) == ("I recently moved.", 0)
    assert resolve_relative_time("yesterday", None) == ("yesterday", 0)
    once, n = resolve_relative_time("yesterday and last week", TS)
    assert n == 2
    assert resolve_relative_time(once, TS) == (once, 0)


def test_longest_match_wins():
    # "last weekend" must not also be read as "last week"
    assert _r("last weekend") == "last weekend (the weekend before 8 May, 2023)"


def _legacy_graph_file(tmpdir):
    """A cache file as written before 2026-09-28: raw text, full timestamp, no meta."""
    import json, os
    data = {"cues": {"Caroline": {"tag_set": ["Support"]}},
            "contents": {"e1": {"text": "Caroline: I went to a support group yesterday.", "layer": "episodic",
                                "time": TS, "tags": ["Support"], "topic_links": []},
                         "s1": {"text": "Caroline values honesty.", "layer": "semantic",
                                "time": None, "tags": ["Support"], "topic_links": []}},
            "links": [["Caroline", "Support", "e1"], ["Caroline", "Support", "s1"]]}
    path = os.path.join(tmpdir, "sample_0.json")
    with open(path, "w") as fh:
        json.dump(data, fh)
    return path


def test_graph_load_resolves_legacy_cache():
    import tempfile
    from iterret.ctc_graph import CueTagContentGraph
    with tempfile.TemporaryDirectory() as d:
        path = _legacy_graph_file(d)
        g = CueTagContentGraph.load(path)
        assert g.contents["e1"].text == "Caroline: I went to a support group yesterday (7 May, 2023)."
        assert g.contents["e1"].time == "8 May, 2023"
        assert g.contents["s1"].text == "Caroline values honesty."          # semantic: untouched
        assert g.meta.get("dates_resolved") is True
        raw = CueTagContentGraph.load(path, resolve_dates=False)             # old behaviour on request
        assert raw.contents["e1"].text.endswith("yesterday.") and raw.contents["e1"].time == TS
        assert not raw.meta.get("dates_resolved")


def test_graph_save_load_does_not_double_annotate():
    import os, tempfile
    from iterret.ctc_graph import CueTagContentGraph
    with tempfile.TemporaryDirectory() as d:
        g = CueTagContentGraph.load(_legacy_graph_file(d))
        out = os.path.join(d, "resaved.json")
        g.save(out)
        again = CueTagContentGraph.load(out)
        assert again.contents["e1"].text == g.contents["e1"].text
        assert again.meta.get("dates_resolved") is True


def test_graph_build_resolves_dates():
    from iterret.llm_client import MockLLMClient
    from iterret.memory_builder import DialogueTurn, build_ctc_graph_from_dialogue
    turns = [DialogueTurn(speaker="Caroline", text="I went to a support group yesterday.", time=TS)]
    g = build_ctc_graph_from_dialogue(turns, MockLLMClient())
    assert g.contents["e1"].text == "Caroline: I went to a support group yesterday (7 May, 2023)."
    assert g.contents["e1"].time == "8 May, 2023" and g.meta.get("dates_resolved") is True
    g_raw = build_ctc_graph_from_dialogue(turns, MockLLMClient(), resolve_dates=False)
    assert g_raw.contents["e1"].text.endswith("yesterday.") and not g_raw.meta.get("dates_resolved")


_ALL_TESTS = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]


def _run_all():
    passed, failed = 0, []
    for fn in _ALL_TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failed.append(fn.__name__)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(_ALL_TESTS)} passed")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
