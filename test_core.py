"""Run: python test_core.py  (no framework; fails with AssertionError)."""
import tempfile

from features import features, parse
from score import observations, score

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
00:03:00.000 --> 00:03:03.000
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


def test_diarized_score_ignores_turns_and_stays_in_range():
    f = {**features(vtt_file()), "diarized": True}
    total, _, parts = score(f)
    assert "student turns per min" not in parts and 0 <= total <= 100
    f["student_turns"] = 10_000  # cluster flips must not move a diarized grade
    assert score(f)[0] == total
    assert all("switches" not in o["point"] for o in observations(parse(vtt_file()), f, {"frames": 0}))


def test_observations_use_real_transcript_lines():
    p = vtt_file()
    f = features(p)
    obs = {o["point"].split()[0]: o for o in observations(parse(p), f, {"frames": 0})}
    assert obs["Opened"]["ok"] and "how are you" in obs["Opened"]["quote"]
    hw = next(o for o in observations(parse(p), f, {"frames": 0}) if "homework" in o["point"].lower())
    assert hw["ok"] and "puzzle three" in hw["quote"]
    dim = observations(parse(p), f, {"frames": 5, "logo_background_share": 0.0, "face_visible_share": 0.1, "face_brightness": 50})
    assert sum(not o["ok"] for o in dim) >= 3  # no background, hidden face, dim face


def test_split_uploads_sorts_by_extension():
    from app import split_uploads
    assert split_uploads(["/x/a.VTT", "/x/b.mp4"]) == ("/x/b.mp4", "/x/a.VTT")
    assert split_uploads(["/x/b.mp4"]) == ("/x/b.mp4", None)
    for bad in ([], ["/x/a.vtt"], ["/x/a.mp4", "/x/b.mp4"]):
        try:
            split_uploads(bad)
            raise AssertionError(bad)
        except Exception as e:
            assert "Upload one" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_split_uploads_sorts_by_extension():
    from app import split_uploads
    assert split_uploads(["/x/a.VTT", "/x/b.mp4"]) == ("/x/b.mp4", "/x/a.VTT")
    assert split_uploads(["/x/b.mp4"]) == ("/x/b.mp4", None)
    for bad in ([], ["/x/a.vtt"], ["/x/a.mp4", "/x/b.mp4"]):
        try:
            split_uploads(bad)
            raise AssertionError(bad)
        except Exception as e:
            assert "Upload one" in str(e)
