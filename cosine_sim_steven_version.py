import argparse
import csv
import json
import os
import glob
import re
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util

# Define the list of MRs that expect a "Difference" input, based on the LLMORPH paper
DIFFERENCE_MRS = {"10", "34", "137", "142", "152"}

# Define the list of MRs that expect an "Equivalence" input, based on the experiment table
EQUIVALENCE_MRS = {"8", "25", "51", "127", "136", "149", "155"}

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


def evaluate_input_equivalence(
    path: str,
    output_base: str = None,
    embedding_name: str = "all-mpnet-base-v2", # Changed from model_name
    batch_size: int = 64,
    ngram_orders: List[int] = None,
    eval_level: str = "gram",
) -> None:
    if ngram_orders is None:
        ngram_orders = [1, 2, 3]

    # 1. Search for JSON files
    json_files = []
    if os.path.isfile(path):
        json_files.append(path)
    elif os.path.isdir(path):
        search_pattern = os.path.join(path, '**', '*.json')
        json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"[Error] No JSON files found at: '{path}'")
        return

    # 2. Load the SentenceTransformer embedding ONCE for efficiency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading SentenceTransformer ({embedding_name}) on {device}...")
    embedding = SentenceTransformer(embedding_name, device=device)

    # =========================================================================
    # DOC LEVEL EVALUATION
    # =========================================================================
    if eval_level == "doc":
        print(f"\n{'='*20} PROCESSING DOC LEVEL {'='*20}")
        grouped_data = defaultdict(list)

        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    json_obj = json.load(f)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping invalid JSON: {file_path}")
                    continue

                data_entries = json_obj.get("data", [])
                if not data_entries:
                    continue

                task_name = json_obj.get("task_name", "unknown_task")
                relation_name = str(json_obj.get("relation_name", "N/A"))
                group_key = (task_name, relation_name)

                for item_idx, item in enumerate(data_entries):
                    shot_id = item.get("id", item_idx)

                    source_input = item.get("source_input", "")
                    if isinstance(source_input, list):
                        source_input = " ".join(source_input)

                    followup_inputs = item.get("followup_inputs", [])

                    for f_idx, f_input in enumerate(followup_inputs):
                        if isinstance(f_input, list):
                            f_input = " ".join(f_input)

                        if not source_input.strip() or not f_input.strip():
                            continue

                        # Append full document inputs directly
                        grouped_data[group_key].append({
                            "shot_id": shot_id,
                            "source_chunk": source_input,
                            "followup_chunk": f_input
                        })

        _process_and_save_groups(
            grouped_data=grouped_data,
            embedding=embedding,
            batch_size=batch_size,
            path=path,
            output_base=output_base,
            embedding_name=embedding_name,
            level_label="doc"
        )

    # =========================================================================
    # GRAM LEVEL EVALUATION
    # =========================================================================
    elif eval_level == "gram":
        for ngram_order in ngram_orders:
            print(f"\n{'='*20} PROCESSING {ngram_order}-GRAM {'='*20}")
            grouped_data = defaultdict(list)

            for file_path in json_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        json_obj = json.load(f)
                    except json.JSONDecodeError:
                        print(f"[Warning] Skipping invalid JSON: {file_path}")
                        continue

                    data_entries = json_obj.get("data", [])
                    if not data_entries:
                        continue

                    task_name = json_obj.get("task_name", "unknown_task")
                    relation_name = str(json_obj.get("relation_name", "N/A"))
                    group_key = (task_name, relation_name)

                    for item_idx, item in enumerate(data_entries):
                        shot_id = item.get("id", item_idx)

                        source_input = item.get("source_input", "")
                        if isinstance(source_input, list):
                            source_input = " ".join(source_input)

                        followup_inputs = item.get("followup_inputs", [])

                        for f_idx, f_input in enumerate(followup_inputs):
                            if isinstance(f_input, list):
                                f_input = " ".join(f_input)

                            if not source_input.strip() or not f_input.strip():
                                continue

                            # Process N-grams
                            src_tokens = tokenize(source_input)
                            fol_tokens = tokenize(f_input)
                            src_ng = get_ngram_strings(src_tokens, ngram_order)
                            fol_ng = get_ngram_strings(fol_tokens, ngram_order)

                            # Pair the N-grams
                            chunk_pairs = pair_ngrams(src_ng, fol_ng)
                            
                            for s_chunk, f_chunk in chunk_pairs:
                                if not s_chunk.strip() or not f_chunk.strip():
                                    continue
                                
                                grouped_data[group_key].append({
                                    "shot_id": shot_id,
                                    "source_chunk": s_chunk,
                                    "followup_chunk": f_chunk
                                })

            if not grouped_data:
                print(f"[Warning] No valid chunk pairs found for {ngram_order}-gram. Skipping...")
                continue

            _process_and_save_groups(
                grouped_data=grouped_data,
                embedding=embedding,
                batch_size=batch_size,
                path=path,
                output_base=output_base,
                embedding_name=embedding_name,
                level_label=f"{ngram_order}gram"
            )


def _process_and_save_groups(
    grouped_data: Dict,
    embedding: SentenceTransformer,
    batch_size: int,
    path: str,
    output_base: str,
    embedding_name: str,
    level_label: str
) -> None:
    """Helper function to compute embeddings and save logs to prevent code duplication."""
    for (task, mr), pairs in grouped_data.items():
        total_pairs = len(pairs)
        print(f"\n[Run] Task: {task} | MR: {mr} | Level: {level_label.upper()} ({total_pairs:,} items)")

        is_difference_mr = mr in DIFFERENCE_MRS
        eval_rule_name = "Difference" if is_difference_mr else "Equivalence"

        # Combine inputs to compute embeddings in bulk, avoiding NxN matrix memory issues
        combined_texts = []
        for p in pairs:
            combined_texts.append(p["source_chunk"])
            combined_texts.append(p["followup_chunk"])
        
        all_embeddings = embedding.encode(combined_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        
        # Split source and followup embeddings
        emb_src = all_embeddings[0::2]
        emb_fol = all_embeddings[1::2]
        
        # Compute element-wise cosine similarity
        raw_scores = torch.nn.functional.cosine_similarity(emb_src, emb_fol).cpu().tolist()

        records = []
        scores_list = []

        for pair, score in zip(pairs, raw_scores):
            cosine_score = float(score)
            scores_list.append(cosine_score)

            records.append({
                "shot_id": pair["shot_id"],
                "cosine_similarity": round(cosine_score, 4),
                "source_text": pair["source_chunk"],
                "followup_text": pair["followup_chunk"]
            })

        mean_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        summary_lines = [
            "=" * 60,
            f"        LLMORPH INPUT TRANSFORMATION EVALUATION SUMMARY",
            "=" * 60,
            f"Timestamp:                 {date_str}",
            f"Task Name:                 {task}",
            f"Relation ID:               {mr}",
            f"Evaluation Rule:           {eval_rule_name}",
            f"Evaluation Level:          {level_label.upper()}",
            f"Mean Cosine Similarity:    {mean_score:.4f}",
            "-" * 60,
            f"Total Evaluated Items:     {total_pairs:,}",
            "=" * 60,
        ]

        print("\n" + "\n".join(summary_lines))

        if output_base:
            project_folder = output_base.strip()
        else:
            abs_path = os.path.abspath(path)
            path_parts = abs_path.split(os.sep)
            project_folder = "llm-nlpaug"
            for part in path_parts:
                if any(k in part for k in ["nlpaug", "transform", "dataset", "results"]):
                    project_folder = part
                    break
            else:
                if len(path_parts) >= 3:
                    project_folder = path_parts[-3]

        output_dir = os.path.join("logs", eval_rule_name, str(mr), project_folder)
        os.makedirs(output_dir, exist_ok=True)

        file_prefix = f"{mr}_input_{eval_rule_name.lower()}_{date_str}_{level_label}"

        # --- Save CSV ---
        csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Shot_ID", "Cosine_Similarity", "Source_Text", "Followup_Text"])
            for r in records:
                writer.writerow([f"#{r['shot_id']}", f"{r['cosine_similarity']:.4f}", r["source_text"], r["followup_text"]])

        # --- Save JSON ---
        json_path = os.path.join(output_dir, f"{file_prefix}.json")
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump({
                "timestamp": date_str,
                "task_name": task,
                "relation_name": mr,
                "embedding_name": embedding_name,
                "rule": eval_rule_name,
                "evaluation_level": level_label,
                "mean_cosine_similarity": round(mean_score, 4),
                "total_items": total_pairs,
                "records": records
            }, f, ensure_ascii=False, indent=4)

        # --- Save TXT Summary ---
        log_path = os.path.join(output_dir, f"{file_prefix}_summary.txt")
        with open(log_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation script for LLM INPUT transformation using cosine similarity.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to evaluated test result JSON file OR directory containing JSON files.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Custom output folder base name.")
    
    # Changed from -m / --model to -e / --embedding
    parser.add_argument("-e", "--embedding", type=str, default="all-mpnet-base-v2", help="Sentence Transformer embedding name.")
    
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for faster inference.")
    
    # Selection for Level: doc or gram
    parser.add_argument("-l", "--level", type=str, choices=["doc", "gram"], default="gram", help="Evaluation level: 'doc' for full input, 'gram' for n-grams.")
    parser.add_argument("-g", "--ngram", nargs="+", type=int, default=[1, 2, 3], help="Specify one or more n-gram orders (e.g., -g 1 2 3). Ignored if level is 'doc'.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_input_equivalence(
        path=args.path,
        output_base=args.output,
        embedding_name=args.embedding, # Passed from args.embedding
        batch_size=args.batch_size,
        ngram_orders=args.ngram,
        eval_level=args.level,
    )
