"""features dict -> 0-100 teaching score and label. Weights hand-set from the reviewer's feedback on 10 samples."""
LABELS = [(70, "Excellent"), (55, "Good"), (40, "Average"), (0, "Poor")]


def clamp(x):
    return max(0.0, min(1.0, x))


def score(f):
    mins = f["duration_min"] or 1
    parts = {
        "student participation (words/min)": 35 * clamp(f["student_words_per_min"] / 50),
        "student turns per min": 20 * clamp(f["student_turns"] / mins / 8),
        "teacher talk share": 15 * (1 - clamp((f["teacher_talk_share"] - 0.5) / 0.3)),
        "rapport intro": 10 * clamp(f["intro_in_first_10pct"]),
        "closing": 8 * clamp(f["bye_in_last_10pct"]),
        "praise": 7 * clamp(f["praise_count"] / 20),
        "homework": 5 * clamp(f["homework_mentions"]),
    }
    total = round(sum(parts.values()))
    return total, next(l for t, l in LABELS if total >= t), parts


if __name__ == "__main__":
    import sys
    from features import features
    for p in sys.argv[1:]:
        t, l, _ = score(features(p))
        print(f"{p.split('/')[-2]:14s} {p.split('/')[-1][:12]:12s} {t:3d} {l}")
