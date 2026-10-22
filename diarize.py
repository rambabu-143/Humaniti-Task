"""Audio -> speaker-labelled cues [(start_sec, speaker, text)], fully local (Phonon-2 words + sherpa-onnx diarization)."""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

M = Path(__file__).parent / "models"


def speakers(wav):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(M / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx")),
            num_threads=4),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(M / "emb.onnx"), num_threads=4),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=4, threshold=0.5),  # ponytail: threshold mode gave 55-166 speakers; 4 ~ teacher + kids, shares within ~0.1 on 2 classes
        min_duration_on=0.3, min_duration_off=0.5)
    audio, _ = sf.read(str(wav), dtype="float32")
    segs = sherpa_onnx.OfflineSpeakerDiarization(cfg).process(audio).sort_by_start_time()
    return [(s.start, s.end, s.speaker) for s in segs]


def words(wav):
    out = subprocess.run([str(Path(__file__).parent / ".venv/bin/phonon"), "transcribe", "--json", str(wav)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out[out.index("{"):])["words"]


def labelled_cues(video):
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(wav)], check=True)
        segs, ws = speakers(wav), words(wav)
    starts = np.array([s[0] for s in segs])
    ends = np.array([s[1] for s in segs])

    def who(w):
        mid = (w["start"] + w["end"]) / 2
        i = int(np.argmin(np.minimum(abs(starts - mid), abs(ends - mid))))  # nearest segment if the word falls in a gap
        return segs[i][2]

    cues, cur = [], None
    for w in ws:
        sp = who(w)
        if cur and cur["sp"] == sp and w["start"] - cur["end"] < 2.0:
            cur["text"] += " " + w["text"]
            cur["end"] = w["end"]
        else:
            cur = {"sp": sp, "start": w["start"], "end": w["end"], "text": w["text"]}
            cues.append(cur)
    return [(int(c["start"]), f"Speaker {c['sp'] + 1}", c["text"]) for c in cues]


if __name__ == "__main__":
    import sys
    for t, s, x in labelled_cues(sys.argv[1])[:25]:
        print(f"{t//60:02d}:{t%60:02d} {s}: {x}")
