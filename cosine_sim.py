import argparse
import csv
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import nltk
import torch
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util


def ensure_wordnet_data() -> None:
    """Downloads NLTK WordNet resources if not already present."""
    for resource in ("corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(resource)
        except LookupError:
            package = resource.split("/")[-1]
            nltk.download(package, quiet=True)


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


def get_wordnet_antonym(word: str) -> str:
    """
    Looks up WordNet to find a true linguistic antonym for a given word chunk.
    Returns the antonym if found, otherwise returns an empty string.
    """
    antonyms = set()
    clean_word = word.strip().lower()
    for syn in wordnet.synsets(clean_word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace('_', ' '))
    if antonyms:
        return list(antonyms)[0]
    return ""


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
        return "Synonym", 0.5   # Synonym threshold: > 0.5 = PASS[cite: 1]
    elif rel_str == "10":
        return "Antonym", 0.5   # Antonym threshold (vs WordNet Antonym): > 0.5 = PASS[cite: 1]
    else:
        print(f"[Warning] Unknown relation_name '{rel_str}', defaulting to Synonym (> 0.5)")
        return "Synonym", 0.5


def evaluate_hybrid_cosine(
    file_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    ngram_order: int = 1,
) -> None:
    """
    Evaluates dataset chunks using Sentence Transformers Cosine Similarity:
    - Synonym: Cosine(Source Chunk, Transformed Chunk) > 0.5[cite: 1]
    - Antonym: Cosine(WordNet Antonym of Source Chunk, Transformed Chunk) > 0.5[cite: 1]
    """
    if not os.path.exists(file_path):
        print(f"[Error] JSON file not found at: '{file_path}'")
        return

    ensure_wordnet_data()

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

    all_evaluated_items = []

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
            if not s_chunk.strip() or not f_chunk.strip():
                continue
            
            if mode == "Synonym":
                if s_chunk.lower() == f_chunk.lower():
                    continue
                all_evaluated_items.append({
                    "shot_id": shot_id,
                    "reference_chunk": s_chunk,      # Compare directly with source
                    "transformed_chunk": f_chunk,
                    "display_source": s_chunk
                })
            else:
                # Antonym Mode: Look up WordNet antonym as ground truth reference (source')[cite: 1]
                wordnet_ant = get_wordnet_antonym(s_chunk)
                if not wordnet_ant:
                    continue
                all_evaluated_items.append({
                    "shot_id": shot_id,
                    "reference_chunk": wordnet_ant,  # Compare with WordNet Antonym (source')[cite: 1]
                    "transformed_chunk": f_chunk,
                    "display_source": s_chunk,
                    "wordnet_antonym": wordnet_ant
                })

    if not all_evaluated_items:
        print("[Error] No valid chunk pairs found for evaluation.")
        return

    print(f"\n[Info] Found a total of {len(all_evaluated_items):,} chunks to evaluate under mode: {mode}.")

    # Deduplicate unique pairs for efficient batch inference
    unique_pairs = list(set((item["reference_chunk"], item["transformed_chunk"]) for item in all_evaluated_items))
    print(f"[Init] Deduplicated to {len(unique_pairs):,} unique pairs for batch inference.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading SentenceTransformer ({model_name}) on {device}...")
    model = SentenceTransformer(model_name, device=device)

    print(f"[Run] Computing Cosine Similarity for unique chunk pairs...")
    
    pair_score_lookup = {}
    valid_pairs = []

    for pair in unique_pairs:
        ref_c, trans_c = pair
        if not ref_c.strip() or not trans_c.strip():
            pair_score_lookup[pair] = 0.0
        else:
            valid_pairs.append(pair)

    if valid_pairs:
        batch_ref = [p[0] for p in valid_pairs]
        batch_trans = [p[1] for p in valid_pairs]
        
        emb_ref = model.encode(batch_ref, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        emb_trans = model.encode(batch_trans, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        raw_scores = util.cos_sim(emb_ref, emb_trans).diagonal().cpu().tolist()

        for pair, sc in zip(valid_pairs, raw_scores):
            pair_score_lookup[pair] = float(sc)

    passed_chunks_count = 0
    records = []

    for item in all_evaluated_items:
        ref_c = item["reference_chunk"]
        trans_c = item["transformed_chunk"]
        score_val = pair_score_lookup.get((ref_c, trans_c), 0.0)

        # Threshold condition: > 0.5 = PASS for both modes[cite: 1]
        is_pass = score_val > threshold

        if is_pass:
            passed_chunks_count += 1

        record_entry = {
            "shot_id": item["shot_id"],
            "status": "PASS" if is_pass else "FAIL",
            "cosine_similarity": round(score_val, 4),
            "source_chunk": item["display_source"],
            "transformed_chunk": trans_c
        }
        if mode == "Antonym":
            record_entry["wordnet_antonym"] = item["wordnet_antonym"]

        records.append(record_entry)

    total_chunks = len(records)
    pass_percentage = (passed_chunks_count / total_chunks * 100.0) if total_chunks > 0 else 0.0

    print("\n================ HYBRID COSINE SIMILARITY RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Evaluation Mode:          {mode}")
    print(f"Threshold (>):            {threshold}")
    print(f"N-gram Order Evaluated:   {ngram_order}-gram")
    print(f"Total Evaluated Chunks:   {total_chunks:,}")
    print(f"Passed Chunks:            {passed_chunks_count:,}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("====================================================================")

    # Output directory structure: logs/{Mode}/{ProjectName}/cosine_chunks_{ngram}gram
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

    output_dir = os.path.join("logs", mode.capitalize(), project_folder, f"cosine_chunks_{ngram_order}gram")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_prefix = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram"

    # Export CSV
    csv_path = os.path.join(output_dir, f"{file_prefix}_chunks.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if mode == "Antonym":
            writer.writerow(["Shot_ID", "Status", "Cosine_Similarity", "Source_Chunk", "WordNet_Antonym", "Transformed_Chunk"])
            for r in records:
                writer.writerow([f"#{r['shot_id']}", r["status"], f"{r['cosine_similarity']:.4f}", r["source_chunk"], r["wordnet_antonym"], r["transformed_chunk"]])
        else:
            writer.writerow(["Shot_ID", "Status", "Cosine_Similarity", "Source_Chunk", "Transformed_Chunk"])
            for r in records:
                writer.writerow([f"#{r['shot_id']}", r["status"], f"{r['cosine_similarity']:.4f}", r["source_chunk"], r["transformed_chunk"]])
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    # Export JSON
    json_path = os.path.join(output_dir, f"{file_prefix}_chunks.json")
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump({
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
            "records": records
        }, f, ensure_ascii=False, indent=4)
    print(f"[Saved] JSON log saved to: '{json_path}'")

    # Export TXT summary
    log_path = os.path.join(output_dir, f"{file_prefix}_chunks_summary.txt")
    summary_lines = [
        "=" * 60,
        f"        HYBRID COSINE SIMILARITY SUMMARY ({mode.upper()})",
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid evaluation script supporting Synonym and WordNet-backed Antonym cosine similarity.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to evaluated test result JSON file.")
    parser.add_argument("-m", "--model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("-g", "--ngram", type=int, default=1, help="Specify n-gram order (e.g., -g 1, -g 2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_hybrid_cosine(
        file_path=args.path,
        model_name=args.model,
        batch_size=args.batch_size,
        ngram_order=args.ngram,
    )
