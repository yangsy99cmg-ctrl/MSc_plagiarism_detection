# inspect_lengths.py
import os, re, json
import pandas as pd
import matplotlib.pyplot as plt

# TRAIN_PATH = r"C:/Users/YangSY/Desktop/homework/MSc_desk/openwebtext_sample.parquet"
TRAIN_PATH = r"C:/Users/YangSY/Desktop/homework/MSc_desk/owt_5_samples/owt_sample_random5pct.parquet"
GEN_PATH   = r"C:/Users/YangSY/Desktop/homework/MSc_desk/GPT_generated_text/pretrainedGPT/large-762M-k40.train.txt"
OUT_DIR    = r"D:/yangsy/results/gpt2_len_probe"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(t: str) -> str:
    if not t: return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_gen_stories_keep_short(gen_text: str, min_chars: int = 200):
    parts = [p.strip() for p in re.split(r"\n{2,}", gen_text) if p.strip()]
    return [p for p in parts if len(p) >= min_chars]

def describe_series(s: pd.Series):
    q = s.quantile([0.1,0.25,0.5,0.75,0.9,0.95,0.99]).to_dict()
    return {
        "count": int(s.size),
        "min": int(s.min()) if s.size else 0,
        "mean": float(s.mean()) if s.size else 0.0,
        "median": int(s.median()) if s.size else 0,
        "max": int(s.max()) if s.size else 0,
        "p10": int(q.get(0.1,0)), "p25": int(q.get(0.25,0)), "p50": int(q.get(0.5,0)),
        "p75": int(q.get(0.75,0)), "p90": int(q.get(0.9,0)), "p95": int(q.get(0.95,0)), "p99": int(q.get(0.99,0)),
    }

def main():
    print("[1] Reading owt_sample_random5pct.parquet …")
    df = pd.read_parquet(TRAIN_PATH)
    if "text" not in df.columns:
        raise ValueError(f"The 'text' column is not found. The actual columns are:{list(df.columns)}")
    df["text"] = df["text"].astype(str).map(clean_text)
    df["char_len"] = df["text"].str.len()
    df_train = df[df["char_len"] > 0].copy()
    stats_train = describe_series(df_train["char_len"])
    df_train[["char_len"]].to_csv(os.path.join(OUT_DIR, "owt_doc_char_lengths.csv"), index=False)

    print("[2] Reading GPT-2 generated text …")
    with open(GEN_PATH, "r", encoding="utf-8", errors="ignore") as f:
        gen_raw = clean_text(f.read())
    stories = split_gen_stories_keep_short(gen_raw, min_chars=200)
    gen_lens = pd.Series([len(s) for s in stories], name="char_len")
    stats_gen = describe_series(gen_lens)
    gen_lens.to_frame().to_csv(os.path.join(OUT_DIR, "gen_story_char_lengths.csv"), index=False)

    print("\n=== OpenWebText Document character count distribution ===")
    for k,v in stats_train.items():
        print(f"{k:>7}: {v}")
    print("\n=== Generate a story with a character count distribution (cut by blank lines, >= 200) ===")
    for k,v in stats_gen.items():
        print(f"{k:>7}: {v}")

    with open(os.path.join(OUT_DIR, "length_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"owt": stats_train, "gen": stats_gen}, f, ensure_ascii=False, indent=2)

    plt.figure()
    df_train["char_len"].clip(upper=df_train["char_len"].quantile(0.99)).hist(bins=50)
    plt.xlabel("OWT doc char length (clipped at P99)")
    plt.ylabel("Count")
    plt.title("OpenWebText document length distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "owt_length_hist.png"), dpi=150)
    plt.close()

    plt.figure()
    gen_lens.clip(upper=gen_lens.quantile(0.99)).hist(bins=50)
    plt.xlabel("Generated story char length (clipped at P99)")
    plt.ylabel("Count")
    plt.title("Generated story length distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gen_length_hist.png"), dpi=150)
    plt.close()

    p40 = int(df_train["char_len"].quantile(0.40))
    p60 = int(df_train["char_len"].quantile(0.60))
    suggestion = {
        "suggest_story_target_range_by_owt_p40_p60": [p40, p60],
        "note": "If you want the search to have the same level of context as the alignment, you can adjust the target length of the generated story to this range (for example, by merging paragraphs)."
    }
    print("\n=== Recommendations (refer to OWT P40–P60) ===")
    print(f"It is recommended to set the target length of generated stories to ~ [{p40}, {p60}] Characters (can be used as merge target intervals)")
    with open(os.path.join(OUT_DIR, "length_suggestion.json"), "w", encoding="utf-8") as f:
        json.dump(suggestion, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Output to:{OUT_DIR}")
    print("  - owt_doc_char_lengths.csv / gen_story_char_lengths.csv")
    print("  - length_summary.json / length_suggestion.json")
    print("  - owt_length_hist.png / gen_length_hist.png")

if __name__ == "__main__":
    main()