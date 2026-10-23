"""Gradio UI: upload a class recording (+ optional .vtt transcript) -> quality grade with evidence."""
import os

import gradio as gr

from pipeline import assess


def run(video, vtt):
    if not video:
        raise gr.Error("Upload a class recording first.")
    r = assess(video, vtt)
    lines = [f"## {r['label']}" + (f"  ({r['score']}/100)" if "score" in r else ""), r.get("summary", "")]
    if r.get("note"):
        lines.append(f"> {r['note']}")
    for title, key in [("Strengths", "strengths"), ("To improve", "improvements")]:
        lines.append(f"### {title}")
        lines += [f"- **{i.get('time', '')}** {i['point']}" + (f"  \n  _\"{i['quote']}\"_" if i.get("quote") else "") for i in r[key]] or ["- (none)"]
    if r.get("tips"):
        lines += ["### Coaching tips"] + [f"- {t}" for t in r["tips"]]
    return "\n".join(lines), r.get("score_breakdown", {}), r["signals"]


demo = gr.Interface(
    run,
    [gr.File(label="Class recording (mp4)", type="filepath"), gr.File(label="Transcript .vtt (optional)", type="filepath")],
    [gr.Markdown(label="Assessment"), gr.JSON(label="Score breakdown"), gr.JSON(label="Raw signals")],
    title="8x8 class quality assessor",
    description="Runs fully locally (Phonon-2, OpenCV, Qwen3-8B). A 50-minute class takes a few minutes.",
    flagging_mode="never",
)

if __name__ == "__main__":
    # local only: recordings show children. Set APP_USER and APP_PASSWORD before exposing it through a tunnel.
    user, pw = os.getenv("APP_USER"), os.getenv("APP_PASSWORD")
    if not (user and pw):
        print("WARNING: APP_USER/APP_PASSWORD not set, so there is no login. Do not expose this through a tunnel.")
    demo.launch(server_name="127.0.0.1", server_port=7860, auth=(user, pw) if user and pw else None)  # fixed port: fail loudly if taken
