"""Run: python test_core.py  (no framework; fails with AssertionError)."""
import tempfile

from features import features, parse
from pipeline import verified
from score import score

VTT = """WEBVTT

1
00:00:05.000 --> 00:00:07.000
Tutor Ann: Hello Ria, how are you?

2
00:00:08.000 --> 00:00:09.000
Ria: Good, ma'am!

3
00:00:10.000 --> 00:00:30.000
Tutor Ann: Today we will learn the knight, and your homework is puzzle three.

4
00:00:31.000 --> 00:00:33.000
Audio shared by Ann: music music music music music music music music music music
"""


def vtt_file():
    f = tempfile.NamedTemporaryFile("w", suffix=".vtt", delete=False)
    f.write(VTT)
    f.close()
    return f.name


def test_parse_and_teacher():
    p = vtt_file()
    assert parse(p)[0] == (5, "Tutor Ann", "Hello Ria, how are you?")
    f = features(p)
    assert f["teacher"] == "Tutor Ann"  # "Audio shared by" must never be the teacher
    assert f["homework_mentions"] == 1 and f["n_students"] == 2


def test_score_orders_participation():
    base = dict(duration_min=45, teacher_talk_share=0.6, student_turns=100, student_words_per_min=20,
                intro_in_first_10pct=1, bye_in_last_10pct=1, praise_count=5, homework_mentions=0)
    lively = {**base, "student_turns": 500, "student_words_per_min": 55}
    silent = {**base, "student_turns": 30, "student_words_per_min": 5, "teacher_talk_share": 0.85}
    assert score(lively)[0] > score(base)[0] > score(silent)[0]
    assert score(lively)[1] == "Excellent" and score(silent)[1] == "Poor"


def test_verified_drops_invented_quotes():
    cues = parse(vtt_file())
    real = {"quote": "Tutor Ann: Hello Ria, how are you?"}
    assert verified([real, {"quote": "You are all doing great"}, "junk"], cues) == [real]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
