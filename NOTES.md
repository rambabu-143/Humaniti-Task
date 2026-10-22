# Findings from the sample data
12 recordings: excellent(2) good(2) average(2) weak(2) quality-issue(2) demo(2, English + Malayalam). 38-90 min each. Every class has a Zoom-style .vtt transcript.
`data/class recordings /feedbacks.pdf` is the human reviewer's per-class comments. This is the rubric:

| Signal | Source | Seen in feedback |
|---|---|---|
| Rapport building / intro at start | transcript (first 3 min) | "directly entered class without rapport" |
| Conclusion / wrap-up | transcript (last 3 min) | "need better intro and conclusion" |
| Individual questions to each child | transcript speaker turns | "asks individually", "considers everyone" |
| New joiners get time to self-introduce | transcript | weak classes |
| Homework given | transcript keywords | "given HWs" / "should give HWs" |
| Energy / pleasant tone | audio variance + LLM on text | "more energetic, pleasant" |
| 8x8 branded background | video frames | "kept 8*8 background" |
| Tutor face visible, lit, camera on | video frames | "can't see his face", "sit where have light" |
| Tutor camera off mid-class | video frames over time | B482 |

"quality issue" folder = tech/setup problems (no background, face hidden), not teaching quality. Treat as a flag, not a fifth class.

Caveats: 2 samples per label, so no training; hand-calibrate rubric weights. A new test video may lack a .vtt, so a Whisper fallback is needed. Demo Malayalam transcript means multilingual input is possible.

# Plan
1. features.py: parse VTT -> talk ratio, per-student turns, questions, HW/intro/conclusion keywords.
2. frames.py: ffmpeg 1 frame/30s -> background/face/lighting checks.
3. LLM rubric scorer -> Poor/Average/Good/Excellent + evidence with timestamps.
4. app.py (Gradio): upload -> result.

Confidentiality: children are visible, so data/ is gitignored. Confirm with the organiser which APIs are allowed before sending frames or transcripts out.
