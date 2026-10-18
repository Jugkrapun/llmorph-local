import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import nltk
import torch
from alignscore import AlignScore
from alignscore.model import BERTAlignModel

# --- Ensure BERTAlignModel.load_from_checkpoint works on instances in PL 2.x ---
_orig_load_from_ckpt = BERTAlignModel.load_from_checkpoint


def _compat_load_from_checkpoint(*args, **kwargs):
    if args and isinstance(args[0], BERTAlignModel):
        instance = args[0]
        args = args[1:]
        if "model" not in kwargs and hasattr(instance, "model"):
            kwargs["model"] = instance.model
    return _orig_load_from_ckpt(*args, **kwargs)


BERTAlignModel.load_from_checkpoint = _compat_load_from_checkpoint
# -------------------------------------------------------------------------------


def extract_texts(item: dict) -> Tuple[str, Optional[str]]:
    """Extracts source and transformed texts from a dataset JSON entry."""
    src_input = item.get("source_input", [])
    src_text = " ".join(src_input) if isinstance(src_input, list) else str(src_input)

    followup_list = item.get("followup_inputs", [])
    if not followup_list or not followup_list[0]:
        return src_text.strip(), None

    fol_target = followup_list[0]
    fol_text = " ".join(fol_target) if isinstance(fol_target, list) else str(fol_target)
    return src_text.strip(), fol_text.strip()


def resolve_mode_by_relation(relation_name: str) -> Tuple[str, float]:
    """Automatically infers evaluation mode and threshold from relation_name."""
    rel_str = str(relation_name).strip()
    if rel_str == "8":
        return "Synonym", 0.7
    elif rel_str == "10":
        return "Antonym", 0.3
    else:
        print(f"[Warning] Unknown relation_name '{rel_str}', defaulting to Synonym (>= 0.7)")
        return "Synonym", 0.7


def evaluate_json_with_alignscore(
    file_path: str,
    ckpt_path: str = "AlignScore-base.ckpt",
    model_name: str = "roberta-base",
    batch_size: int = 32,
) -> None:
    if not os.path.exists(file_path):
        print(f"[Error] JSON file not found at: '{file_path}'")
        return

    # Download required NLTK tokenizers directly without triggering OSError in nltk.data.find
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

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

    shot_ids: List[int] = []
    contexts: List[str] = []
    claims: List[str] = []

    for idx, item in enumerate(data_entries):
        shot_id = item.get("id", idx)
        src_text, fol_text = extract_texts(item)
        if not src_text or not fol_text:
            continue
        shot_ids.append(shot_id)
        contexts.append(src_text)
        claims.append(fol_text)

    if not contexts:
        print("[Error] No valid text pairs found for evaluation.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[Init] Loading AlignScore ({model_name}) on {device}...")

    scorer = AlignScore(
        model=model_name,
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
    )

    print(f"[Run] Computing AlignScore for {len(contexts)} shots...")
    scores = scorer.score(contexts=contexts, claims=claims)

    passed_count = 0
    records: List[Dict] = []

    for shot_id, src, fol, score in zip(shot_ids, contexts, claims, scores):
        score_val = float(score)
        is_pass = (score_val >= threshold) if mode == "Synonym" else (score_val <= threshold)
        if is_pass:
            passed_count += 1

        records.append({
            "shot_id": shot_id,
            "status": "PASS" if is_pass else "FAIL",
            "source_text": src,
            "transformed_text": fol,
            "align_score": round(score_val, 4),
        })

    pass_percentage = (passed_count / len(records) * 100.0) if records else 0.0

    print("\n================ ALIGNSCORE EVALUATION RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Mode:                     {mode} (Threshold: {threshold})")
    print(f"Total Evaluated Shots:    {len(records)}")
    print(f"Passed Threshold:         {passed_count}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("===============================================================")

    output_dir = os.path.join("logs", mode.capitalize(), "alignscore")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_alignscore.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Shot_ID", "Status", "AlignScore", "Source_Text", "Transformed_Text"])
        for r in records:
            writer.writerow([
                f"#{r['shot_id']}",
                r["status"],
                f"{r['align_score']:.4f}",
                r["source_text"],
                r["transformed_text"],
            ])

    print(f"[Saved] CSV results saved to: '{csv_path}'")

    log_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_alignscore_summary.txt"
    log_path = os.path.join(output_dir, log_filename)

    summary_lines = [
        "=" * 60,
        "          ALIGNSCORE EVALUATION SUMMARY LOG",
        "=" * 60,
        f"Timestamp:                 {date_str}",
        f"Model Evaluated:           {llm_name}",
        f"Task Name:                 {task_name}",
        f"Relation ID:               {relation_name}",
        f"Evaluation Mode:           {mode}",
        f"Score Threshold:           {threshold}",
        f"Backbone Model:            {model_name}",
        "-" * 60,
        f"Total Evaluated Shots:     {len(records):,}",
        f"Passed Threshold:          {passed_count:,}",
        f"Pass Percentage:           {pass_percentage:.2f}%",
        "=" * 60,
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"[Saved] Summary log saved to: '{log_path}'")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JSON dataset using AlignScore metric.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the evaluated test result JSON file.")
    parser.add_argument("-c", "--ckpt", type=str, default="AlignScore-base.ckpt", help="Path to checkpoint (.ckpt).")
    parser.add_argument("-m", "--model", type=str, default="roberta-base", help="Backbone model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_json_with_alignscore(
        file_path=args.path,
        ckpt_path=args.ckpt,
        model_name=args.model,
        batch_size=args.batch_size,
    )
