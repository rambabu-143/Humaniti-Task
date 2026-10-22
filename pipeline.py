"""Recording (+ optional .vtt) -> {label, observations, signals}. Fully local: Phonon-2, OpenCV, Qwen3-8B via llama-server."""
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from features import features, parse
from frames import frame_features
from score import score

LLM = "http://localhost:8080/v1/chat/completions"
LABELS = ["Poor", "Average", "Good", "Excellent"]

RUBRIC = """You grade online kids' chess classes run by the 8x8 academy (Eight Times Eight) on the criteria its human reviewers use.
Positive: warm rapport-building intro (asks kids how they are, small talk) before content; a proper closing (recap, thanks, goodbye);
asking questions to kids INDIVIDUALLY by name so everyone is included; kids talking a lot (high student participation);
giving homework; energetic, pleasant, encouraging tone with praise; 8x8 branded background; tutor's face visible and well lit.
Negative: jumping straight into revision with no rapport; no introduction time for new joiners; flat/low-energy tone; teacher monologue;
ignoring some kids; no homework; no 8x8 background; face hidden/dark; tutor camera off.
Calibration from reviewed examples: Excellent classes had ~500+ student turns and 30-60 student words/min, teacher talk share 50-70%, named students addressed often.
Poor classes had teacher talk share 74-80%, <20 student words/min, weak intro/closing. Note: camera/background issues are setup issues; they lower a grade by at most one level.
Signals are automatic and noisy (keyword counts, face detector): treat them as hints, ground every observation in the transcript."""


def transcribe(video):
    """No .vtt supplied: Phonon-2 locally. ponytail: no speaker labels, so no talk-ratio features; add diarization later."""
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(wav)], check=True)
        out = subprocess.run(["phonon", "transcribe", "--json", str(wav)], capture_output=True, text=True, check=True).stdout
    segs = json.loads(out[out.index("{"):]).get("segments", [])
    return [(int(s.get("start", 0)), "?", s.get("text", "").strip()) for s in segs]


def excerpt(cues, dur):
    """Intro + outro + a slice of the middle, as 'mm:ss Speaker: text' (keeps the prompt ~6k tokens)."""
    def block(lo, hi):
        return "\n".join(f"{t//60:02d}:{t%60:02d} {s}: {x}" for t, s, x in cues if lo <= t < hi)
    mid = dur / 2
    return (f"[START]\n{block(0, 360)}\n[MIDDLE]\n{block(mid - 180, mid + 180)}\n[END]\n{block(dur - 300, dur + 1)}")[:22000]


def ask(prompt):
    body = {"messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt + " /no_think"}],
            "temperature": 0.2, "response_format": {"type": "json_object"}}
    req = urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"])


def norm(t):
    return " ".join("".join(c if c.isalnum() else " " for c in t.lower()).split())


def verified(items, cues):
    """Keep only observations whose quote really appears in the transcript."""
    text = norm(" ".join(f"{c[1]}: {c[2]}" for c in cues) + " " + " ".join(c[2] for c in cues))
    return [i for i in items if isinstance(i, dict) and i.get("quote") and norm(i["quote"]) in text]


def assess(video, vtt=None, cache="cache/frames"):
    video = Path(video)
    cues = parse(vtt) if vtt else transcribe(video)
    feats = features(vtt) if vtt else {}
    fr = frame_features(video, Path(cache) / video.stem.replace(" ", ""), cues if vtt else None, feats.get("teacher"))
    dur = max(c[0] for c in cues)
    fixed = ""
    if feats:
        total, label, parts = score(feats)
        fixed = f"The grade is already decided: {label} (score {total}/100). Explain it; do not change it.\nScore breakdown: {json.dumps({k: round(v) for k, v in parts.items()})}\n"
    prompt = (f"{fixed}Automatic signals:\n{json.dumps(feats)}\nCamera/background signals (tutor-speaking frames only; share 0-1):\n{json.dumps(fr)}\n\n"
              f"Transcript excerpts:\n{excerpt(cues, dur)}\n\n"
              'Return JSON: {"label": "Poor|Average|Good|Excellent", "summary": "2 sentences", '
              '"strengths": [{"point": "...", "quote": "EXACT words copied from the transcript", "time": "mm:ss"}], '
              '"improvements": [{"point": "...", "quote": "EXACT words copied from the transcript", "time": "mm:ss"}], '
              '"setup_flags": []}. setup_flags: list a flag only when the camera signals show a real problem (face_visible_share under 0.2, no background, brightness under 70); otherwise leave it empty')
    res = ask(prompt)
    res["strengths"] = verified(res.get("strengths", []), cues)
    res["improvements"] = verified(res.get("improvements", []), cues)
    if feats:
        res.update(label=label, score=total, score_breakdown={k: round(v) for k, v in parts.items()})
    else:
        res["note"] = "No speaker labels (no .vtt): grade is the model's judgement from text only, less reliable."
        if res.get("label") not in LABELS:
            res["label"] = "Average"  # ponytail: model strayed from the label set
    return {**res, "signals": {**feats, **fr}}


if __name__ == "__main__":
    import sys
    v = sys.argv[1]
    vtt = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(assess(v, vtt), indent=2, default=float))
