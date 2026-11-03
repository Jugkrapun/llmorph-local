import argparse
import csv
import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from sentence_transformers import SentenceTransformer, util


def evaluate_output_equivalence(
    path: str,
    output_base: str = None,
    model_name: str = "paraphrase-MiniLM-L6-v2",
    batch_size: int = 64,
) -> None:
    """
    Evaluates LLM Output Equivalence using Sentence Transformers.
    - Matches `source_output` vs `followup_outputs` at the document level (full text).
    - Equivalence Threshold >= 0.8
    """
    # 1. Search for JSON files (supports both single file and directory paths)
    json_files = []
    if os.path.isfile(path):
        json_files.append(path)
    elif os.path.isdir(path):
        search_pattern = os.path.join(path, '**', '*.json')
        json_files = glob.glob(search_pattern, recursive=True)
    
    if not json_files:
        print(f"[Error] No JSON files found at: '{path}'")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading SentenceTransformer ({model_name}) on {device}...")
    model = SentenceTransformer(model_name, device=device)

    llm_name = "unknown_model"
    task_name = "unknown_task"
    relation_name = "N/A"
    
    collected_pairs = []

    # 2. Read data and prepare sentence pairs for computation
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
            
            # Extract metadata from the first successfully read file
            if llm_name == "unknown_model":
                llm_name = json_obj.get("llm_name", "unknown_model")
                task_name = json_obj.get("task_name", "question_answering")
                relation_name = str(json_obj.get("relation_name", "N/A"))

            for item_idx, item in enumerate(data_entries):
                shot_id = item.get("id", item_idx)
                
                # Extract Source Output
                source_output = item.get("source_output", "")
                if isinstance(source_output, list):
                    source_output = " ".join(source_output)
                
                # Extract Follow-up Outputs
                followup_outputs = item.get("followup_outputs", [])
                
                for f_idx, f_output in enumerate(followup_outputs):
                    if isinstance(f_output, list):
                        f_output = " ".join(f_output)
                    
                    if not source_output.strip() or not f_output.strip():
                        continue

                    collected_pairs.append({
                        "shot_id": shot_id,
                        "source_output": source_output,
                        "followup_output": f_output
                    })

    total_pairs = len(collected_pairs)
    if total_pairs == 0:
        print("[Error] No valid output pairs found for evaluation.")
        return

    print(f"[Run] Computing Cosine Similarity for {total_pairs:,} pairs in batches...")

    # 3. Compute Cosine Similarity in batches for processing speed
    batch_src = [p["source_output"] for p in collected_pairs]
    batch_fol = [p["followup_output"] for p in collected_pairs]
    
    emb_src = model.encode(batch_src, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    emb_fol = model.encode(batch_fol, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    
    raw_scores = util.cos_sim(emb_src, emb_fol).diagonal().cpu().tolist()

    # 4. Evaluate equivalence based on LLMORPH's threshold (>= 0.8)
    records = []
    passed_count = 0
    scores_list = []

    for pair, score in zip(collected_pairs, raw_scores):
        cosine_score = float(score)
        scores_list.append(cosine_score)
        
        is_pass = cosine_score >= 0.8
        if is_pass:
            passed_count += 1
            
        records.append({
            "shot_id": pair["shot_id"],
            "status": "PASS" if is_pass else "FAIL",
            "cosine_similarity": round(cosine_score, 4),
            "source_output": pair["source_output"],
            "followup_output": pair["followup_output"]
        })

    pass_percentage = (passed_count / total_pairs * 100.0)
    mean_score = sum(scores_list) / len(scores_list)

    print("\n================ LLMORPH OUTPUT EQUIVALENCE RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Embedding Model:          {model_name}")
    print(f"Mean Cosine Similarity:   {mean_score:.4f}")
    print(f"Equivalence Threshold:    >= 0.8")
    print(f"Total Evaluated Pairs:    {total_pairs:,}")
    print(f"Passed (Equivalent):      {passed_count:,}")
    print(f"Equivalence Percentage:   {pass_percentage:.2f}%")
    print("====================================================================")

    # 5. Prepare the output directory for saving files
    if output_base:
        project_folder = output_base.strip()
    else:
        abs_path = os.path.abspath(path)
        path_parts = abs_path.split(os.sep)
        project_folder = "llm-nlpaug"
        for part in path_parts:
            if "nlpaug" in part or "transform" in part or "dataset" in part or "results" in part:
                project_folder = part
                break
        else:
            if len(path_parts) >= 3:
                project_folder = path_parts[-3]

    output_dir = os.path.join("logs", "Equivalence", project_folder)
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_prefix = f"{llm_name}_rel{relation_name}_equiv_{date_str}"

    # --- Save CSV file ---
    csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Shot_ID", "Status", "Cosine_Similarity", "Source_Output", "Followup_Output"])
        for r in records:
            writer.writerow([f"#{r['shot_id']}", r["status"], f"{r['cosine_similarity']:.4f}", r["source_output"], r["followup_output"]])
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    # --- Save JSON log ---
    json_path = os.path.join(output_dir, f"{file_prefix}.json")
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump({
            "timestamp": date_str,
            "model_evaluated": llm_name,
            "task_name": task_name,
            "relation_name": relation_name,
            "embedding_model": model_name,
            "score_threshold": 0.8,
            "mean_cosine_similarity": round(mean_score, 4),
            "total_pairs": total_pairs,
            "passed_pairs": passed_count,
            "equivalence_percentage": round(pass_percentage, 2),
            "records": records
        }, f, ensure_ascii=False, indent=4)
    print(f"[Saved] JSON log saved to: '{json_path}'")

    # --- Save TXT Summary file ---
    log_path = os.path.join(output_dir, f"{file_prefix}_summary.txt")
    summary_lines = [
        "=" * 60,
        f"        LLMORPH OUTPUT EQUIVALENCE SUMMARY",
        "=" * 60,
        f"Timestamp:                 {date_str}",
        f"Model Evaluated:           {llm_name}",
        f"Task Name:                 {task_name}",
        f"Relation ID:               {relation_name}",
        f"Embedding Model:           {model_name}",
        f"Equivalence Threshold:     >= 0.8",
        f"Mean Cosine Similarity:    {mean_score:.4f}",
        "-" * 60,
        f"Total Evaluated Pairs:     {total_pairs:,}",
        f"Passed (Equivalent):       {passed_count:,}",
        f"Equivalence Percentage:    {pass_percentage:.2f}%",
        "=" * 60,
    ]
    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Saved] Summary log saved to: '{log_path}'")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation script for LLM output equivalence using cosine similarity.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to evaluated test result JSON file OR directory containing JSON files.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Custom output folder base name.")
    parser.add_argument("-m", "--model", type=str, default="paraphrase-MiniLM-L6-v2", help="Sentence Transformer model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for faster inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_output_equivalence(
        path=args.path,
        output_base=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
    )
