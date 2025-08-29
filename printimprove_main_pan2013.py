# main_pan2013.py
import os
import json
from sklearn.metrics import classification_report

from do_language_detector import detect_and_classify
from utils import load_doc, save_json


def get_ground_truth_mapping(pan_root):
    type_map = {
        "01-no-plagiarism": "none",
        "02-no-obfuscation": "verbatim",
        "03-random-obfuscation": "verbatim",
        "04-translation-obfuscation": "paraphrase",
        "05-summary-obfuscation": "summary"
    }
    ground_truth = {}
    for folder, label in type_map.items():
        folder_path = os.path.join(pan_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".xml"):
                continue
            parts = fname.replace(".xml", "").split("-source-document")
            susp = parts[0] + ".txt"
            src = "source-document" + parts[1] + ".txt"
            ground_truth[(susp, src)] = label
    return ground_truth


def main():
    root = "C:/Users/YangSY/Desktop/homework/MSc_desk/pan13-text-alignment-test-corpus2-2013-01-21"
    susp_dir = os.path.join(root, "susp")
    src_dir = os.path.join(root, "src")
    pairs_file = os.path.join(root, "pairs")
    out_dir = "C:/Users/YangSY/Desktop/homework/MSc_desk/ALL_Results/pan2013_outputs"
    os.makedirs(out_dir, exist_ok=True)

    ground_truth_map = get_ground_truth_mapping(root)
    all_preds, all_labels = [], []

    with open(pairs_file, "r") as f:
        pairs = [line.strip().split() for line in f]

    for susp, src in pairs:
        susp_path = os.path.join(susp_dir, susp)
        src_path  = os.path.join(src_dir,  src)
        print(f"\nProcessing: {susp} <-> {src}")

        label_type = ground_truth_map.get((susp, src), "none")

        classified_segments = detect_and_classify(
            susp_path, src_path, out_dir,
            low_alignment=1,
            min_char_cov=200,
            sent_min_sim=0.85,
            para_min_sim=0.75,   
            para_topk=5,         
            min_match_words=5,
        )

        num_segs = len(classified_segments)
        types_count = {}
        for seg in classified_segments:
            t = seg.get("plagiarism_type", "none")
            types_count[t] = types_count.get(t, 0) + 1

        if not classified_segments:
            pred_type = "none"
        else:
            sums = {"verbatim": 0, "paraphrase": 0, "summary": 0}
            for seg in classified_segments:
                t = seg["plagiarism_type"]
                if t == "none":
                    continue
                a, b = seg["source_offset"], seg["suspicious_offset"]
                length = max(a[1] - a[0], b[1] - b[0])
                if t in sums:
                    sums[t] += length
            pred_type = max(sums, key=sums.get) if any(sums.values()) else "none"

        print(f"  -> Detected {num_segs} segments {types_count} | Pred={pred_type} | GT={label_type}")

        result_path = os.path.join(
            out_dir, f"{os.path.splitext(susp)[0]}--{os.path.splitext(src)[0]}.json"
        )
        save_json(classified_segments, result_path)

        all_preds.append(pred_type)
        all_labels.append(label_type)


    print("\nEvaluation Results:")
    print(classification_report(all_labels, all_preds))
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(
            classification_report(all_labels, all_preds, output_dict=True),
            f, indent=2
        )


if __name__ == "__main__":
    main()