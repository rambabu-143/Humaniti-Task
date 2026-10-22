"""Transcript (.vtt) -> numeric teaching-quality signals."""
import re
import sys
from collections import Counter
from pathlib import Path

CUE = re.compile(r"(\d+):(\d+):(\d+)\.\d+ --> (\d+):(\d+):(\d+)\.\d+\n(.+?)(?:\n\n|\Z)", re.S)
HW = re.compile(r"homework|home work|\bHW\b|assignment|practice at home", re.I)
INTRO = re.compile(r"introduce yourself|your name|tell me about yourself|how are you|how was your|welcome", re.I)
BYE = re.compile(r"see you|bye|thank you|good job|well done|next class|that's all", re.I)
PRAISE = re.compile(r"good job|well done|excellent|great|very good|nice|awesome|perfect|wonderful|super", re.I)


def parse(path):
    """-> list of (start_sec, speaker, text)."""
    out = []
    for m in CUE.finditer(Path(path).read_text(encoding="utf-8-sig").replace("\r", "")):
        h, mi, s = int(m[1]), int(m[2]), int(m[3])
        line = " ".join(m[7].split())
        speaker, _, text = line.partition(": ")
        if not text:
            speaker, text = "?", line
        out.append((h * 3600 + mi * 60 + s, speaker, text))
    return out


def features(path):
    cues = parse(path)
    dur = max(c[0] for c in cues) or 1
    words = Counter()
    for _, sp, t in cues:
        words[sp] += len(t.split())
    teacher = words.most_common(1)[0][0]  # ponytail: teacher = most words; fails if kids dominate
    students = [s for s in words if s != teacher]
    t_cues = [c for c in cues if c[1] == teacher]
    first, last = dur * 0.1, dur * 0.9
    total = sum(words.values()) or 1
    return {
        "teacher": teacher,
        "duration_min": round(dur / 60, 1),
        "teacher_talk_share": round(words[teacher] / total, 2),
        "n_students": len(students),
        "student_turns": sum(1 for c in cues if c[1] != teacher),
        "student_words_per_min": round((total - words[teacher]) / (dur / 60), 1),
        "teacher_questions": sum(c[2].count("?") for c in t_cues),
        "praise_count": sum(len(PRAISE.findall(c[2])) for c in t_cues),
        "homework_mentions": sum(bool(HW.search(c[2])) for c in t_cues),
        "intro_in_first_10pct": sum(bool(INTRO.search(c[2])) for c in t_cues if c[0] < first),
        "bye_in_last_10pct": sum(bool(BYE.search(c[2])) for c in t_cues if c[0] > last),
        "named_students_addressed": sum(
            any(s.split()[0].lower() in c[2].lower() for s in students) for c in t_cues
        ),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p.split("/")[-2], p.split("/")[-1][:20], features(p))
