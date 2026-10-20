import argparse
import csv
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util


def tokenize(text: str) -> List[str]:
    """Tokenizes input text into a list of words using regular expressions."""
    return re.findall(r"\w+", text)


def get_ngram_strings(tokens: List[str], n: int) -> List[str]:
    """Generates sliding-window n-gram strings of size n from a token list."""
    if not tokens:
        return []
    if len(tokens) < n:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def pair_ngrams(src_ngrams: List[str], fol_ngrams: List[str]) -> List[Tuple[str, str]]:
    """
    Pairs source and followup n-grams using SequenceMatcher opcodes:
    - 'equal': Ignored (no transformation).
    - 'replace': Handled via direct 1:1 or proportional mapping.
    - 'delete'/'insert': Handled via empty string mapping for strict penalization.
    """
    if not src_ngrams or not fol_ngrams:
        return []

    matcher = SequenceMatcher(None, src_ngrams, fol_ngrams)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            s_range = list(range(i1, i2))
            f_range = list(range(j1, j2))
            max_len = max(len(s_range), len(f_range))
            for i in range(max_len):
                s_idx = s_range[min(i, len(s_range) - 1)]
                f_idx = f_range[min(i, len(f_range) - 1)]
                pairs.append((src_ngrams[s_idx], fol_ngrams[f_idx]))
        elif tag == 'delete':
            for s_idx in range(i1, i2):
                pairs.append((src_ngrams[s_idx], ""))
        elif tag == 'insert':
            for f_idx in range(j1, j2):
                pairs.append(("", fol_ngrams[f_idx]))

    return pairs


def resolve_mode_by_relation(relation_name: str) -> Tuple[str, float]:
    """Automatically infers the evaluation mode and threshold based on relation ID."""
    rel_str = str(relation_name).strip()
    if rel_str == "8":
        return "Synonym", 0.5   # Synonym threshold: > 0.5 = PASS
    elif rel_str == "10":
        return "Antonym", -0.5  # Antonym threshold: < -0.5 = PASS
    else:
        print(f"[Warning] Unknown relation_name '{rel_str}', defaulting to Synonym (> 0.5)")
        return "Synonym", 0.5


def evaluate_chunks_cosine(
    file_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    ngram_order: int = 1,
) -> None:
    """
    Extracts all changed/inserted/deleted n-gram chunks across the dataset,
    evaluates them via Sentence Transformers Cosine Similarity, and computes pass percentage.
    """
    # Verify file existence
    if not os.path.exists(file_path):
        print(f"[Error] JSON file not found at: '{file_path}'")
        return

    # Load dataset JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)

    data_entries = json_obj.get("data", [])
    if not data_entries:
        print("[Error] No entries found in the JSON 'data' field.")
        return

    llm_name = json_obj.get("llm_name", "unknown_model")
    task_name = json_obj.get("task_name", "question_answering")
    relation_name = str(json_obj.get("relation_name", "N/A"))
    mode, threshold = resolve_mode_by_relation(relation_name)

    all_extracted_chunks = []

    # Iterate through dataset entries to extract chunk modifications
    for idx, item in enumerate(data_entries):
        shot_id = item.get("id", idx)
        src_input = item.get("source_input", [])
        src_text = " ".join(src_input) if isinstance(src_input, list) else str(src_input)

        followup_list = item.get("followup_inputs", [])
        if not followup_list or not followup_list[0]:
            continue
        fol_target = followup_list[0]
        fol_text = " ".join(fol_target) if isinstance(fol_target, list) else str(fol_target)

        src_tokens = tokenize(src_text)
        fol_tokens = tokenize(fol_text)
        src_ng = get_ngram_strings(src_tokens, ngram_order)
        fol_ng = get_ngram_strings(fol_tokens, ngram_order)

        pairs = pair_ngrams(src_ng, fol_ng)
        for s_chunk, f_chunk in pairs:
            # Skip empty strings or identical text cases (case-insensitive)
            if s_chunk.strip() and f_chunk.strip() and s_chunk.lower() == f_chunk.lower():
                continue
            all_extracted_chunks.append({
                "shot_id": shot_id,
                "source_chunk": s_chunk,
                "transformed_chunk": f_chunk
            })

    if not all_extracted_chunks:
        print("[Error] No modified n-gram chunks found for evaluation.")
        return

    print(f"\n[Info] Found a total of {len(all_extracted_chunks):,} evaluated chunks.")

    # Deduplicate unique pairs of chunks for efficient batch inference
    unique_pairs = list(set((c["source_chunk"], c["transformed_chunk"]) for c in all_extracted_chunks))
    print(f"[Init] Deduplicated to {len(unique_pairs):,} unique chunk pairs for batch inference.")

    # Initialize computation device and load SentenceTransformer model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading SentenceTransformer ({model_name}) on {device}...")
    model = SentenceTransformer(model_name, device=device)

    print(f"[Run] Computing Cosine Similarity for unique chunk pairs...")
    
    pair_score_lookup = {}
    valid_pairs = []

    # Filter out empty strings before inference
    for pair in unique_pairs:
        s_c, f_c = pair
        if not s_c.strip() or not f_c.strip():
            pair_score_lookup[pair] = 0.0
        else:
            valid_pairs.append(pair)

    if valid_pairs:
        batch_src = [p[0] for p in valid_pairs]
        batch_fol = [p[1] for p in valid_pairs]
        
        # Batch encode sentences and compute cosine similarity diagonal matrix
        emb_src = model.encode(batch_src, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        emb_fol = model.encode(batch_fol, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        raw_scores = util.cos_sim(emb_src, emb_fol).diagonal().cpu().tolist()

        for pair, sc in zip(valid_pairs, raw_scores):
            pair_score_lookup[pair] = float(sc)

    passed_chunks_count = 0
    records = []

    # Evaluate each record against the established threshold
    for item in all_extracted_chunks:
        s_chunk = item["source_chunk"]
        f_chunk = item["transformed_chunk"]
        score_val = pair_score_lookup.get((s_chunk, f_chunk), 0.0)

        # Check conditions based on mode: Synonym (> threshold) or Antonym (< threshold)
        if mode == "Synonym":
            is_pass = score_val > threshold
        else:
            is_pass = score_val < threshold

        if is_pass:
            passed_chunks_count += 1

        records.append({
            "shot_id": item["shot_id"],
            "status": "PASS" if is_pass else "FAIL",
            "cosine_similarity": round(score_val, 4),
            "source_chunk": s_chunk,
            "transformed_chunk": f_chunk
        })

    total_chunks = len(records)
    pass_percentage = (passed_chunks_count / total_chunks * 100.0) if total_chunks > 0 else 0.0

    # Print summary output to console
    print("\n================ CHUNK-LEVEL COSINE SIMILARITY RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Mode:                     {mode} (Threshold: {threshold})")
    print(f"N-gram Order Evaluated:   {ngram_order}-gram")
    print(f"Total Evaluated Chunks:   {total_chunks:,}")
    print(f"Passed Chunks:            {passed_chunks_count:,}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("========================================================================")

    # Automatically structure output directory path: logs/{Mode}/{ProjectName}/cosine
    abs_path = os.path.abspath(file_path)
    path_parts = abs_path.split(os.sep)
    project_folder = "llm-nlpaug"
    for part in path_parts:
        if "nlpaug" in part or "transform" in part or "dataset" in part:
            project_folder = part
            break
    else:
        if len(path_parts) >= 3:
            project_folder = path_parts[-3]

    output_dir = os.path.join("logs", mode.capitalize(), project_folder, "cosine")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Export results to CSV file
    csv_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Shot_ID", "Status", "Cosine_Similarity", "Source_Chunk", "Transformed_Chunk"])
        for r in records:
            writer.writerow([
                f"#{r['shot_id']}",
                r["status"],
                f"{r['cosine_similarity']:.4f}",
                r["source_chunk"],
                r["transformed_chunk"],
            ])
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    # Export summary log to TXT file
    log_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram_summary.txt"
    log_path = os.path.join(output_dir, log_filename)

    summary_lines = [
        "=" * 60,
        f"        CHUNK-LEVEL {ngram_order}-GRAM COSINE SIMILARITY SUMMARY",
        "=" * 60,
        f"Timestamp:                 {date_str}",
        f"Model Evaluated:           {llm_name}",
        f"Task Name:                 {task_name}",
        f"Relation ID:               {relation_name}",
        f"Evaluation Mode:           {mode}",
        f"Score Threshold:           {threshold}",
        f"N-gram Order:              {ngram_order}",
        "-" * 60,
        f"Total Evaluated Chunks:    {total_chunks:,}",
        f"Passed Chunks:             {passed_chunks_count:,}",
        f"Pass Percentage:           {pass_percentage:.2f}%",
        "=" * 60,
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Saved] Summary log saved to: '{log_path}'")

    # Export structured results to JSON file
    json_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram.json"
    json_path = os.path.join(output_dir, json_filename)

    json_output = {
        "timestamp": date_str,
        "model_evaluated": llm_name,
        "task_name": task_name,
        "relation_name": relation_name,
        "evaluation_mode": mode,
        "score_threshold": threshold,
        "ngram_order": ngram_order,
        "total_chunks": total_chunks,
        "passed_chunks": passed_chunks_count,
        "pass_percentage": round(pass_percentage, 2),
        "records": records,
    }

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
    print(f"[Saved] JSON log saved to: '{json_path}'")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for evaluation configuration."""
    parser = argparse.ArgumentParser(description="Evaluate chunks using Sentence Transformers Cosine Similarity.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the evaluated test result JSON file.")
    parser.add_argument("-m", "--model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("-g", "--ngram", type=int, default=1, help="Specify n-gram order (e.g., -g 1, -g 2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_chunks_cosine(
        file_path=args.path,
        model_name=args.model,
        batch_size=args.batch_size,
        ngram_order=args.ngram,
    )
