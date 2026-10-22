# 8x8 class quality assessor

Upload an online class recording (and its `.vtt` transcript if you have one). Get **Poor / Average / Good / Excellent** plus evidence-backed observations. Everything runs locally: recordings show children and never leave the machine.

## How it works
1. **Transcript**: uses the `.vtt` if given (has speaker names). Otherwise Phonon-2 transcribes the audio locally (no speaker labels, so the grade is less reliable).
2. **Signals** (`features.py`): student words/min, student turns, teacher talk share, rapport intro, closing, praise, homework.
3. **Camera tile** (`frames.py`): every 30s, only frames where the tutor is speaking: 8x8 logo, face visible, face brightness.
4. **Grade** (`score.py`): hand-set weights, taken from the human reviewer's feedback on the sample classes. The label comes from this score, not the LLM.
5. **Explanation** (`pipeline.py`): Qwen3-8B writes strengths/improvements. Each must quote the transcript; quotes that don't appear in it are dropped.

## Run
Needs macOS/Linux, Python 3.10+, `ffmpeg`, and [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`).

```
python3 -m venv .venv && .venv/bin/pip install "fermion-research[mlx]" "opencv-python-headless<5" gradio
# terminal 1: local LLM
llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M -c 16384 --port 8080 --jinja
# terminal 2: app (http://127.0.0.1:7860)
.venv/bin/python app.py
```
Check: `.venv/bin/python test_core.py`. A 50-minute class takes 1-5 minutes.

## Limitations
- Weights are fitted by hand on 10 labelled classes (2 per grade): a rubric, not a trained model.
- Without a `.vtt` there is no teacher/student split; the strongest signal is lost.
- The logo check assumes the Zoom speaker tile sits top-right; it missed the background in 3 classes the reviewer said had one.
- Mostly-English; Malayalam classes are unreliable.
