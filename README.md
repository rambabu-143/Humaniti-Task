# 8x8 class quality assessor

Upload an online class recording (and its `.vtt` transcript if you have one). Get **Poor / Average / Good / Excellent** plus evidence-backed observations. Everything runs locally: recordings show children and never leave the machine.

## How it works
1. **Transcript**: uses the `.vtt` if given (has speaker names). Otherwise Phonon-2 transcribes locally and sherpa-onnx separates speakers into 4 clusters (teacher = most words); per-kid signals are skipped and the grade is flagged lower confidence.
2. **Signals** (`features.py`): student words/min, student turns, teacher talk share, rapport intro, closing, praise, homework.
3. **Camera tile** (`frames.py`): every 30s, only frames where the tutor is speaking: 8x8 logo, face visible, face brightness.
4. **Grade** (`score.py`): hand-set weights, taken from the human reviewer's feedback on the sample classes. The label comes from this score, not the LLM.
5. **Explanation** (`score.py`, `pipeline.py`): each rubric item is checked in code and backed by the real transcript line or number. Qwen3-8B only writes the summary and coaching tips.

## Run
Needs an Apple-silicon Mac (Phonon-2 runs on MLX), Python 3.10+, `ffmpeg`, and [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`).

```
python3 -m venv .venv && .venv/bin/pip install "fermion-research[mlx]" "opencv-python-headless<5" gradio sherpa-onnx soundfile numpy
# (or: uv venv && uv pip install <same packages>)
# speaker-separation models, needed only when no .vtt is given (not in git, ~50 MB)
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 && tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -L -o emb.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx
cd ..
# terminal 1: local LLM
llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M -c 16384 --port 8080 --jinja
# terminal 2: app (http://127.0.0.1:7860)
.venv/bin/python app.py
```
In the app, drop the recording (and its `.vtt` if you have one) into the single upload box.
Tests: `.venv/bin/python test_core.py` (no pytest needed; `pytest` also works).

### Sharing a demo link (recordings stay on your machine)
```
export APP_USER=demo APP_PASSWORD='<pick one>'   # login for the page
.venv/bin/python app.py                          # terminal 2, as above (llama-server must be running)
cloudflared tunnel --url http://127.0.0.1:7860   # terminal 3: prints a https://*.trycloudflare.com link
```
Share the link and password privately (not in the repo). The Mac must stay on and awake while it is tested; stop the tunnel with Ctrl+C when done. A 50-minute class takes 1-5 minutes.

## Limitations
- Weights are fitted by hand on the 8 reviewer-labelled classes (2 each of excellent, good, average, weak): a rubric, not a trained model, and the accuracy numbers below are on the same classes (not held-out).
- Without a `.vtt` the teacher/student split comes from diarization. On the 8 labelled classes it gets 4 exact grades and 7 within one level (with a `.vtt`: 6 exact); weak B558 is graded Excellent because the teacher's speech is split into "student" clusters. Treat it as approximate.
- The logo and face checks are whole-frame template/Haar matches sampled every 30s; they agree with the reviewer on all 10 English classes, but chat-shared homework or background changes outside the transcript can't be seen.
- With a `.vtt`, 6 of the 8 labelled classes get the exact grade (the others are one level off: B435 average -> Good, B558 weak -> Average).
- Mostly-English; Malayalam classes are unreliable.
