"""Recording (+ optional .vtt) -> {label, observations, signals}. Fully local: Phonon-2, OpenCV, Qwen3-8B via llama-server."""
import json
import urllib.request
from pathlib import Path

from diarize import labelled_cues
from features import features, parse
from frames import frame_features
from score import observations, score

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


def ask(prompt):
    body = {"messages": [{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt + " /no_think"}],
            "temperature": 0.2, "response_format": {"type": "json_object"}}
    req = urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req, timeout=900).read())["choices"][0]["message"]["content"])


def assess(video, vtt=None, cache="cache/frames"):
    video = Path(video)
    if vtt:
        cues = parse(vtt)
        feats = features(cues)
    else:
        cues = labelled_cues(video)  # local diarization: speaker identity is noisy, so score() skips per-kid signals
        feats = {**features(cues), "diarized": True}
    fr = frame_features(video, Path(cache) / video.stem.replace(" ", ""), cues, feats["teacher"])
    total, label, parts = score(feats)
    obs = observations(cues, feats, fr)
    prompt = (f"Grade: {label} ({total}/100). Rubric findings (already verified against the transcript):\n"
              + "\n".join(f"- {'OK' if o['ok'] else 'ISSUE'}: {o['point']}" for o in obs)
              + '\nReturn JSON: {"summary": "2 sentences explaining the grade using only these findings", '
                '"tips": ["up to 3 concrete coaching tips for the tutor, one sentence each"]}')
    res = ask(prompt)
    res.update(label=label, score=total, score_breakdown={k: round(v) for k, v in parts.items()},
               strengths=[o for o in obs if o["ok"]], improvements=[o for o in obs if not o["ok"]])
    if feats.get("diarized"):
        res["note"] = "No .vtt: speakers were separated automatically, so the teacher/student split is approximate and per-kid signals are skipped. Lower confidence."
    return {**res, "signals": {**feats, **fr}}


if __name__ == "__main__":
    import sys
    v = sys.argv[1]
    vtt = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(assess(v, vtt), indent=2, default=float))
