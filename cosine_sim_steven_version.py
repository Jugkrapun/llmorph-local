import argparse
import csv
import json
import os
import glob
from datetime import datetime
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer, util

# Define the list of MRs that expect a "Difference" input, based on the LLMORPH paper
DIFFERENCE_MRS = {"10", "34", "137", "142", "152"}

def evaluate_input_equivalence(
    path: str,
    output_base: str = None,
    model_name: str = "paraphrase-MiniLM-L6-v2", # Can be changed to "all-mpnet-base-v2" when calling
) -> None:
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

    # 2. Read and group data by (LLM, Task, MR) to prevent metadata conflicts
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
            
            llm_name = json_obj.get("llm_name", "unknown_model")
            task_name = json_obj.get("task_name", "unknown_task")
            relation_name = str(json_obj.get("relation_name", "N/A"))
            
            group_key = (llm_name, task_name, relation_name)

            for item_idx, item in enumerate(data_entries):
                shot_id = item.get("id", item_idx)
                
                # Extract source input data
                source_input = item.get("source_input", "")
                if isinstance(source_input, list):
                    source_input = " ".join(source_input)
                
                followup_inputs = item.get("followup_inputs", [])
                
                for f_idx, f_input in enumerate(followup_inputs):
                    if isinstance(f_input, list):
                        f_input = " ".join(f_input)
                    
                    if not source_input.strip() or not f_input.strip():
                        continue

                    grouped_data[group_key].append({
                        "shot_id": shot_id,
                        "source_input": source_input,
                        "followup_input": f_input
                    })

    if not grouped_data:
        print("[Error] No valid input pairs found for evaluation.")
        return

    # 3. Load the SentenceTransformer model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading SentenceTransformer ({model_name}) on {device}...")
    model = SentenceTransformer(model_name, device=device)

    # 4. Process and evaluate results group by group
    for (llm, task, mr), pairs in grouped_data.items():
        total_pairs = len(pairs)
        print(f"\n[Run] Computing embeddings pairwise for {llm} | Task: {task} | MR: {mr}...")

        is_difference_mr = mr in DIFFERENCE_MRS
        eval_rule_name = "Difference" if is_difference_mr else "Equivalence"

        records = []
        scores_list = []

        # Process and compute scores pairwise
        for pair in pairs:
            embeddings = model.encode([pair["source_input"], pair["followup_input"]], convert_to_tensor=True, show_progress_bar=False)
            cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            cosine_score = float(cosine_sim)
            
            scores_list.append(cosine_score)
            
            records.append({
                "shot_id": pair["shot_id"],
                "cosine_similarity": round(cosine_score, 4),
                "source_input": pair["source_input"],
                "followup_input": pair["followup_input"]
            })

        mean_score = sum(scores_list) / len(scores_list) if scores_list else 0.0
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 5. Build the summary layout
        summary_lines = [
            "=" * 60,
            f"        LLMORPH INPUT TRANSFORMATION EVALUATION SUMMARY",
            "=" * 60,
            f"Timestamp:                 {date_str}",
            f"Model Evaluated:           {llm}",
            f"Task Name:                 {task}",
            f"Relation ID:               {mr}",
            f"Evaluation Rule:           {eval_rule_name}",
            f"Mean Cosine Similarity:    {mean_score:.4f}",
            "-" * 60,
            f"Total Evaluated Pairs:     {total_pairs:,}",
            "=" * 60,
        ]

        print("\n" + "\n".join(summary_lines))

        # 6. Save logs and outputs separated by group
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

        # Structure: logs/Difference/10/project_folder
        output_dir = os.path.join("logs", eval_rule_name, str(mr), project_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        file_prefix = f"{llm}_rel{mr}_input_{eval_rule_name.lower()}_{date_str}"

        # --- Save CSV ---
        csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Shot_ID", "Cosine_Similarity", "Source_Input", "Followup_Input"])
            for r in records:
                writer.writerow([f"#{r['shot_id']}", f"{r['cosine_similarity']:.4f}", r["source_input"], r["followup_input"]])

        # --- Save JSON ---
        json_path = os.path.join(output_dir, f"{file_prefix}.json")
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump({
                "timestamp": date_str,
                "model_evaluated": llm,
                "task_name": task,
                "relation_name": mr,
                "embedding_model": model_name,
                "rule": eval_rule_name,
                "mean_cosine_similarity": round(mean_score, 4),
                "total_pairs": total_pairs,
                "records": records
            }, f, ensure_ascii=False, indent=4)

        # --- Save TXT Summary ---
        log_path = os.path.join(output_dir, f"{file_prefix}_summary.txt")
        with open(log_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation script for LLM INPUT transformation equivalence/difference using cosine similarity.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to evaluated test result JSON file OR directory containing JSON files.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Custom output folder base name.")
    parser.add_argument("-m", "--model", type=str, default="paraphrase-MiniLM-L6-v2", help="Sentence Transformer model name.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    evaluate_input_equivalence(
        path=args.path,
        output_base=args.output,
        model_name=args.model,
    )
