"""features dict -> 0-100 teaching score and label. Weights hand-set from the reviewer's feedback on 10 samples."""
LABELS = [(70, "Excellent"), (55, "Good"), (40, "Average"), (0, "Poor")]


def clamp(x):
    return max(0.0, min(1.0, x))


def score(f):
    mins = f["duration_min"] or 1
    parts = {
        "student participation (words/min)": 35 * clamp(f["student_words_per_min"] / 50),
        "student turns per min": 20 * clamp(f["student_turns"] / mins / 8),  # dropped when diarized: turns are cluster flips, not real turns
        "teacher talk share": 15 * (1 - clamp((f["teacher_talk_share"] - 0.5) / 0.3)),
        "rapport intro": 10 * clamp(f["intro_in_first_10pct"]),
        "closing": 8 * clamp(f["bye_in_last_10pct"]),
        "praise": 7 * clamp(f["praise_count"] / 20),
        "homework": 5 * clamp(f["homework_mentions"]),
    }
    if f.get("diarized"):
        del parts["student turns per min"]
        parts = {k: v * 100 / 80 for k, v in parts.items()}
    total = round(sum(parts.values()))
    return total, next(l for t, l in LABELS if total >= t), parts


if __name__ == "__main__":
    import sys
    from features import features
    for p in sys.argv[1:]:
        t, l, _ = score(features(p))
        print(f"{p.split('/')[-2]:14s} {p.split('/')[-1][:12]:12s} {t:3d} {l}")


def observations(cues, f, fr):
    """Rubric checks with evidence taken straight from the transcript -> [{ok, point, time, quote}]."""
    import re
    from features import BYE, HW, INTRO, PRAISE
    t = f["teacher"]
    dur = max(c[0] for c in cues)
    mine = [c for c in cues if c[1] == t]

    def first(rx, lo=0, hi=1e9):
        return next((c for c in mine if lo <= c[0] <= hi and rx.search(c[2])), None)

    def ev(ok, point, c=None, quote=None):
        ts = f"{c[0] // 60:02d}:{c[0] % 60:02d}" if c else ""
        return {"ok": ok, "point": point, "time": ts, "quote": (f"{c[1]}: {c[2]}"[:160] + ("…" if len(c[2]) > 150 else "")) if c else quote}

    out = []
    c = first(INTRO, 0, dur * 0.1)
    out.append(ev(bool(c), "Opened with rapport-building" if c else "No rapport-building in the opening; went straight to content",
                  c or mine[0]))
    c = first(BYE, dur * 0.9)
    out.append(ev(bool(c), "Closed the class properly" if c else "No proper closing or goodbye in the last minutes",
                  c or mine[-1]))
    c = first(HW)
    out.append(ev(bool(c), "Gave homework" if c else "Did not give homework", c))
    names = {s.split()[0].lower() for _, s, _ in cues if s != t and not s.startswith(("Audio", "?")) and len(s.split()[0]) > 2}
    sw = f["kid_switches_per_min"]
    named = next((c for c in mine if "?" in c[2] and any(w.strip(",.?!").lower() in names for w in c[2].split())), None)
    if not f.get("diarized"):
        out.append(ev(sw >= 2.5, f"Involved many different kids in turn ({sw} switches between kids/min)" if sw >= 2.5
                      else f"Few kids took turns ({sw} switches between kids/min); call on each child in turn", named))
    swpm = f["student_words_per_min"]
    out.append(ev(swpm >= 30, f"Kids spoke a lot ({swpm} words/min, {f['teacher_talk_share']:.0%} teacher talk)" if swpm >= 30
                  else f"Kids spoke little ({swpm} words/min; teacher talk {f['teacher_talk_share']:.0%}); ask more, talk less",
                  quote=f"{f['student_turns']} student turns in {f['duration_min']} min"))
    pr = [c for c in mine if PRAISE.search(c[2])]
    out.append(ev(len(pr) >= 10, f"Encouraging tone with praise ({len(pr)} times)" if len(pr) >= 10
                  else f"Little praise ({len(pr)} times); a more energetic, pleasant tone would help", pr[0] if pr else None))
    run = start = 0
    where = None
    for c in cues:
        if c[1] == t and not c[1].startswith("Audio"):
            start = start or c[0]
            if c[0] - start > run:
                run, where = c[0] - start, start
        else:
            start = 0
    if run > 240:
        w = next(c for c in cues if c[0] == where)
        out.append(ev(False, f"Teacher spoke alone for {run // 60} min without kids replying", w))
    if fr["frames"]:
        if fr["logo_background_share"] < 0.1:
            out.append(ev(False, "8x8 branded background not visible", quote=f"logo seen in {fr['logo_background_share']:.0%} of tutor frames"))
        if fr["face_visible_share"] < 0.2:
            out.append(ev(False, "Tutor's face rarely visible (camera off/hidden)", quote=f"face seen in {fr['face_visible_share']:.0%} of tutor frames"))
        if fr["face_brightness"] and fr["face_brightness"] < 70:
            out.append(ev(False, "Tutor's face is dim; sit where there is light", quote=f"face brightness {fr['face_brightness']:.0f}/255"))
    return out
