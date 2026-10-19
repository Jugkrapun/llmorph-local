import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import nltk
import spacy
import torch
from alignscore import AlignScore
import alignscore.inference as align_inf
from alignscore.model import BERTAlignModel


# --- Monkey-patch AlignScore Inferencer to support PyTorch Lightning 2.x ---
def _patched_inferencer_init(
    self,
    ckpt_path,
    model="bert-base-uncased",
    batch_size=32,
    device="cuda",
    verbose=True,
):
    """Patches AlignScore Inferencer initialization to be compatible with PyTorch Lightning 2.x."""
    self.device = device
    if ckpt_path is not None:
        self.model = BERTAlignModel.load_from_checkpoint(
            checkpoint_path=ckpt_path, model=model, strict=False
        ).to(self.device)
    else:
        self.model = BERTAlignModel(model=model).to(self.device)
    self.model.eval()
    self.batch_size = batch_size
    self.spacy = spacy.load("en_core_web_sm")
    self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    self.softmax = torch.nn.Softmax(dim=-1)
    self.smart_type = "smart-n"
    self.smart_n_metric = "f1"
    self.disable_progress_bar_in_inference = False
    self.verbose = verbose


align_inf.Inferencer.__init__ = _patched_inferencer_init
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Tokenizes input text into a list of lowercase words."""
    return re.findall(r"\w+", text.lower())


def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Generates n-grams of size n from the token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def calculate_ngram_metrics(src_text: str, fol_text: str, n_values: List[int]) -> Dict[str, float]:
    """Computes Jaccard similarity-based overlap scores for specified n-gram sizes."""
    src_tokens = tokenize(src_text)
    fol_tokens = tokenize(fol_text)

    metrics = {}
    for n in n_values:
        src_ngrams = get_ngrams(src_tokens, n)
        fol_ngrams = get_ngrams(fol_tokens, n)

        if not src_ngrams or not fol_ngrams:
            metrics[f"{n}_gram"] = 0.0
            continue

        src_counter = Counter(src_ngrams)
        fol_counter = Counter(fol_ngrams)

        # Calculate intersection and union for n-gram frequency counters
        intersection = sum((src_counter & fol_counter).values())
        union = sum((src_counter | fol_counter).values())

        score = (intersection / union) if union > 0 else 0.0
        metrics[f"{n}_gram"] = round(score, 4)

    return metrics


def extract_texts(item: dict) -> Tuple[str, Optional[str]]:
    """Extracts source input text and transformed followup text from a dataset JSON entry."""
    src_input = item.get("source_input", [])
    src_text = " ".join(src_input) if isinstance(src_input, list) else str(src_input)

    followup_list = item.get("followup_inputs", [])
    if not followup_list or not followup_list[0]:
        return src_text.strip(), None

    fol_target = followup_list[0]
    fol_text = " ".join(fol_target) if isinstance(fol_target, list) else str(fol_target)
    return src_text.strip(), fol_text.strip()


def resolve_mode_by_relation(relation_name: str) -> Tuple[str, float]:
    """Automatically infers the evaluation mode (Synonym/Antonym) and score threshold based on relation ID."""
    rel_str = str(relation_name).strip()
    if rel_str == "8":
        return "Synonym", 0.7
    elif rel_str == "10":
        return "Antonym", 0.3
    else:
        print(f"[Warning] Unknown relation_name '{rel_str}', defaulting to Synonym (>= 0.7)")
        return "Synonym", 0.7


def evaluate_json_with_alignscore_and_ngrams(
    file_path: str,
    ckpt_path: str = "AlignScore-base.ckpt",
    model_name: str = "roberta-base",
    batch_size: int = 32,
    ngram_orders: List[int] = [1, 2, 3],
) -> None:
    """Main function to evaluate dataset JSON using hybrid AlignScore and custom N-gram metrics."""
    if not os.path.exists(file_path):
        print(f"[Error] JSON file not found at: '{file_path}'")
        return

    # Download required NLTK tokenizer resources safely
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
    mode, align_threshold = resolve_mode_by_relation(relation_name)

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

    # Initialize AlignScore scorer
    scorer = AlignScore(
        model=model_name,
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
    )

    print(f"[Run] Computing AlignScore & {ngram_orders}-gram metrics for {len(contexts)} shots...")
    align_scores = scorer.score(contexts=contexts, claims=claims)

    passed_count = 0
    records: List[Dict] = []

    for shot_id, src, fol, align_sc in zip(shot_ids, contexts, claims, align_scores):
        align_val = float(align_sc)
        
        # Calculate custom n-gram metrics based on user specification (-g)
        ngram_metrics = calculate_ngram_metrics(src, fol, ngram_orders)
        ngram_values_list = list(ngram_metrics.values())
        avg_ngram = sum(ngram_values_list) / len(ngram_values_list) if ngram_values_list else 0.0

        # Define PASS/FAIL criteria combining AlignScore and local N-gram lexical overlap
        if mode == "Synonym":
            is_pass = (align_val >= align_threshold) or (avg_ngram >= 0.6)
        else:
            is_pass = (align_val <= align_threshold)

        if is_pass:
            passed_count += 1

        record_entry = {
            "shot_id": shot_id,
            "status": "PASS" if is_pass else "FAIL",
            "align_score": round(align_val, 4),
        }
        for k, v in ngram_metrics.items():
            record_entry[k] = v
        record_entry["avg_ngram"] = round(avg_ngram, 4)
        record_entry["source_text"] = src
        record_entry["transformed_text"] = fol

        records.append(record_entry)

    pass_percentage = (passed_count / len(records) * 100.0) if records else 0.0

    print("\n================ HYBRID ALIGNSCORE + N-GRAM RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Mode:                     {mode} (Align Threshold: {align_threshold})")
    print(f"N-gram Orders Evaluated:  {ngram_orders}")
    print(f"Total Evaluated Shots:    {len(records)}")
    print(f"Passed Threshold:         {passed_count}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("====================================================================")

    output_dir = os.path.join("logs", mode.capitalize(), "alignscore_hybrid")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # CSV Headers dynamically include requested n-grams
    csv_headers = ["Shot_ID", "Status", "AlignScore"] + [f"{n}_gram" for n in ngram_orders] + ["Avg_Ngram", "Source_Text", "Transformed_Text"]

    csv_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_hybrid.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(csv_headers)
        for r in records:
            row_data = [
                f"#{r['shot_id']}",
                r["status"],
                f"{r['align_score']:.4f}",
            ]
            for n in ngram_orders:
                row_data.append(f"{r.get(f'{n}_gram', 0.0):.4f}")
            row_data.extend([
                f"{r['avg_ngram']:.4f}",
                r["source_text"],
                r["transformed_text"],
            ])
            writer.writerow(row_data)
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    # 2. Export summary log to TXT file
    log_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_hybrid_summary.txt"
    log_path = os.path.join(output_dir, log_filename)

    summary_lines = [
        "=" * 60,
        "       HYBRID ALIGNSCORE + N-GRAM EVALUATION SUMMARY",
        "=" * 60,
        f"Timestamp:                 {date_str}",
        f"Model Evaluated:           {llm_name}",
        f"Task Name:                 {task_name}",
        f"Relation ID:               {relation_name}",
        f"Evaluation Mode:           {mode}",
        f"Align Threshold:           {align_threshold}",
        f"N-gram Orders:             {ngram_orders}",
        "-" * 60,
        f"Total Evaluated Shots:     {len(records):,}",
        f"Passed Threshold:          {passed_count:,}",
        f"Pass Percentage:           {pass_percentage:.2f}%",
        "=" * 60,
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Saved] Summary log saved to: '{log_path}'")

    # 3. Export complete data to JSON file
    json_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_hybrid.json"
    json_path = os.path.join(output_dir, json_filename)

    json_output = {
        "timestamp": date_str,
        "model_evaluated": llm_name,
        "task_name": task_name,
        "relation_name": relation_name,
        "evaluation_mode": mode,
        "align_threshold": align_threshold,
        "ngram_orders": ngram_orders,
        "total_shots": len(records),
        "passed_threshold": passed_count,
        "pass_percentage": round(pass_percentage, 2),
        "records": records,
    }

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
    print(f"[Saved] JSON log saved to: '{json_path}'")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate JSON dataset using Hybrid AlignScore + Custom N-gram metrics.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the evaluated test result JSON file.")
    parser.add_argument("-c", "--ckpt", type=str, default="AlignScore-base.ckpt", help="Path to checkpoint (.ckpt).")
    parser.add_argument("-m", "--model", type=str, default="roberta-base", help="Backbone model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("-g", "--ngram", type=int, nargs="+", default=[1, 2, 3], help="Specify n-gram orders to calculate (e.g., -g 1 2 3 or -g 1 2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_json_with_alignscore_and_ngrams(
        file_path=args.path,
        ckpt_path=args.ckpt,
        model_name=args.model,
        batch_size=args.batch_size,
        ngram_orders=args.ngram,
    )
