# tests/test_pan_unit.py
import os, io, json, sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main_pan2013 import get_ground_truth_mapping
from do_language_detector import detect_and_classify

def _read_json(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_ground_truth_mapping_basic(tiny_pan):
    gt = get_ground_truth_mapping(str(tiny_pan))
    assert len(gt) == 3
    assert gt.get(("suspicious-document00001.txt", "source-document00001.txt")) == "verbatim"
    assert gt.get(("suspicious-document00002.txt", "source-document00002.txt")) == "paraphrase"
    assert gt.get(("suspicious-document00003.txt", "source-document00003.txt")) == "summary"

def test_detect_and_classify_smoke(tiny_pan, out_dir, monkeypatch):
    import do_language_detector as dl
    monkeypatch.setattr(dl, "bm25_realign_sentences",  lambda *a, **k: [])
    monkeypatch.setattr(dl, "bm25_realign_paragraphs", lambda *a, **k: [])
    monkeypatch.setattr(dl, "run_text_alignment_loose", lambda *a, **k: [])

    pairs = [
        ("suspicious-document00001.txt", "source-document00001.txt"),
        ("suspicious-document00002.txt", "source-document00002.txt"),
        ("suspicious-document00003.txt", "source-document00003.txt"),
    ]
    susp_dir = os.path.join(str(tiny_pan), "susp")
    src_dir  = os.path.join(str(tiny_pan), "src")

    for susp, src in pairs:
        susp_path = os.path.join(susp_dir, susp)
        src_path  = os.path.join(src_dir,  src)

        segs = detect_and_classify(
            susp_path, src_path, str(out_dir),
            low_alignment=1, min_char_cov=50,
            sent_min_sim=0.85, para_min_sim=0.75, para_topk=3, min_match_words=3
        )
        assert isinstance(segs, list)

        out_name = f"{os.path.splitext(susp)[0]}--{os.path.splitext(src)[0]}.json"
        out_path = os.path.join(str(out_dir), out_name)
        with io.open(out_path, "w", encoding="utf-8") as f:
            json.dump(segs, f, ensure_ascii=False, indent=2)

        assert os.path.exists(out_path), f"缺少输出：{out_path}"
        with io.open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)

def test_main_loop_like_processing(tiny_pan, out_dir, monkeypatch, capsys):
    """
    Emulate the main loop of main_pan2013.py (but using tiny data):
      - Read pairs
      - Run detect_and_classify (disable callbacks)
      - Print a summary for each pair
    Purpose: Verify end-to-end crash-free operation, correct I/O, and readable print format.
    """
    import do_language_detector as dl
    monkeypatch.setattr(dl, "bm25_realign_sentences", lambda *a, **k: [])
    monkeypatch.setattr(dl, "bm25_realign_paragraphs", lambda *a, **k: [])
    monkeypatch.setattr(dl, "run_text_alignment_loose", lambda *a, **k: [])

    pairs = [
        ("suspicious-document00001.txt", "source-document00001.txt"),
        ("suspicious-document00002.txt", "source-document00002.txt"),
        ("suspicious-document00003.txt", "source-document00003.txt"),
    ]
    with io.open(os.path.join(str(tiny_pan), "pairs"), "w", encoding="utf-8") as f:
        for s, r in pairs:
            f.write(f"{s} {r}\n")

    gt = get_ground_truth_mapping(str(tiny_pan))
    susp_dir = os.path.join(str(tiny_pan), "susp")
    src_dir  = os.path.join(str(tiny_pan), "src")

    preds, labels = [], []

    for susp, src in pairs:
        susp_path = os.path.join(susp_dir, susp)
        src_path  = os.path.join(src_dir,  src)
        print(f"\nProcessing: {susp} <-> {src}")

        segs = dl.detect_and_classify(
            susp_path, src_path, str(out_dir),
            low_alignment=1, min_char_cov=50,
            sent_min_sim=0.85, para_min_sim=0.75, para_topk=3, min_match_words=3
        )

        if not segs:
            pred_type = "none"
        else:
            sums = {"verbatim":0, "paraphrase":0, "summary":0}
            for seg in segs:
                t = seg.get("plagiarism_type", "none")
                if t == "none":
                    continue
                a, b = seg["source_offset"], seg["suspicious_offset"]
                length = max(a[1]-a[0], b[1]-b[0])
                if t in sums:
                    sums[t] += length
            pred_type = max(sums, key=sums.get) if any(sums.values()) else "none"

        gt_type = gt.get((susp, src), "none")
        print(f"  -> Detected {len(segs)} segments | Pred={pred_type} | GT={gt_type}")

        preds.append(pred_type); labels.append(gt_type)

    out = capsys.readouterr().out
    assert "Processing:" in out