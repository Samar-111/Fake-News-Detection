import re
import string
import json
import numpy as np
import pandas as pd
from pathlib import Path

def extract_source(text):
    if not isinstance(text, str):
        return "Unknown"
    reuters_match = re.search(r'\((Reuters|reuters)\)', text)
    if reuters_match:
        return "Reuters"
    ap_match = re.search(r'\b(AP|Associated Press)\b', text[:150])
    if ap_match:
        return "Associated Press"
    bbc_match = re.search(r'\b(BBC)\b', text[:150])
    if bbc_match:
        return "BBC"
    cnn_match = re.search(r'\b(CNN)\b', text[:150])
    if cnn_match:
        return "CNN"
    return "Independent / Web Source"

def clean_text_content(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def prepare_unified_dataset(
    raw_path_fn="data/raw/fake_and_real_news.csv",
    raw_path_gfn="data/raw/GlobalFakeNews.csv",
    raw_path_extra="data/raw/extra_fake.csv",
    output_path="data/processed/cleaned_news_11k.csv",
    summary_path="data/processed/eda_summary.json"
):
    dfs = []
    
    if Path(raw_path_fn).exists():
        df_fn = pd.read_csv(raw_path_fn)
        df_fn = df_fn.rename(columns={"Text": "text"})
        df_fn["label"] = df_fn["label"].astype(str).str.lower().map({"fake": 1, "real": 0})
        df_fn["title"] = df_fn["text"].apply(lambda t: str(t)[:90].strip() if pd.notna(t) else "")
        df_fn["source_name"] = df_fn["text"].apply(extract_source)
        df_fn["publish_date"] = pd.date_range(start="2023-01-01", periods=len(df_fn), freq="45min").strftime("%Y-%m-%d")
        df_fn["origin"] = "ISOT_Subset"
        dfs.append(df_fn[["title", "text", "source_name", "publish_date", "label", "origin"]])

    if Path(raw_path_gfn).exists():
        df_gfn = pd.read_csv(raw_path_gfn)
        df_gfn = df_gfn.rename(columns={"full_text": "text"})
        df_gfn["label"] = df_gfn["label"].astype(int)
        df_gfn["source_name"] = df_gfn["source_name"].fillna("Independent Publisher")
        df_gfn["publish_date"] = pd.to_datetime(df_gfn["publish_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df_gfn["publish_date"] = df_gfn["publish_date"].fillna("2024-06-01")
        df_gfn["origin"] = "GlobalResearch"
        dfs.append(df_gfn[["title", "text", "source_name", "publish_date", "label", "origin"]])

    if Path(raw_path_extra).exists():
        df_extra = pd.read_csv(raw_path_extra, nrows=2500)
        df_extra = df_extra.rename(columns={"text": "text"})
        df_extra["label"] = 1
        df_extra["title"] = df_extra["text"].apply(lambda t: str(t)[:90].strip() if pd.notna(t) else "")
        df_extra["source_name"] = "Alternative Wire / Blog"
        df_extra["publish_date"] = pd.date_range(start="2024-01-01", periods=len(df_extra), freq="2h").strftime("%Y-%m-%d")
        df_extra["origin"] = "ExtraFake"
        dfs.append(df_extra[["title", "text", "source_name", "publish_date", "label", "origin"]])

    combined = pd.concat(dfs, ignore_index=True)
    initial_count = len(combined)
    
    combined = combined.dropna(subset=["text", "label"])
    combined["text"] = combined["text"].apply(clean_text_content)
    combined = combined[combined["text"].str.len() > 20]
    
    duplicate_count = combined.duplicated(subset=["text"]).sum()
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    combined["char_count"] = combined["text"].apply(len)
    combined["word_count"] = combined["text"].apply(lambda t: len(t.split()))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    combined.to_csv(str(output_path) + ".gz", index=False, compression="gzip")
    
    label_counts = combined["label"].value_counts().to_dict()
    source_counts = combined["source_name"].value_counts().head(12).to_dict()
    
    summary = {
        "initial_records": int(initial_count),
        "final_records": int(len(combined)),
        "duplicates_removed": int(duplicate_count),
        "class_distribution": {
            "real_articles": int(label_counts.get(0, 0)),
            "fake_articles": int(label_counts.get(1, 0)),
            "imbalance_ratio": round(float(label_counts.get(1, 0)) / max(1, float(label_counts.get(0, 0))), 4)
        },
        "length_statistics": {
            "avg_char_count": round(float(combined["char_count"].mean()), 2),
            "median_char_count": round(float(combined["char_count"].median()), 2),
            "avg_word_count": round(float(combined["word_count"].mean()), 2),
            "median_word_count": round(float(combined["word_count"].median()), 2),
            "min_word_count": int(combined["word_count"].min()),
            "max_word_count": int(combined["word_count"].max())
        },
        "top_publishers": source_counts
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    return combined, summary

if __name__ == "__main__":
    df, s = prepare_unified_dataset()
    print("Prepared dataset successfully with shape:", df.shape)
    print("Summary:", json.dumps(s, indent=2))
