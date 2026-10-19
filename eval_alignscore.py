import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase words."""
    return re.findall(r"\w+", text.lower())


def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Generates n-grams for a given n."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def calculate_ngram_metrics(src_text: str, fol_text: str) -> Dict[str, float]:
    """Computes overlap scores for 1-gram, 2-gram, and 3-gram."""
    src_tokens = tokenize(src_text)
    fol_tokens = tokenize(fol_text)

    metrics = {}
    for n in range(1, 4):
        src_ngrams = get_ngrams(src_tokens, n)
        fol_ngrams = get_ngrams(fol_tokens, n)

        if not src_ngrams or not fol_ngrams:
            metrics[f"{n}_gram"] = 0.0
            continue

        src_counter = Counter(src_ngrams)
        fol_counter = Counter(fol_ngrams)

        # Intersection over Union (Jaccard Similarity) for n-grams
        intersection = sum((src_counter & fol_counter).values())
        union = sum((src_counter | fol_counter).values())

        score = (intersection / union) if union > 0 else 0.0
        metrics[f"{n}_gram"] = round(score, 4)

    return metrics


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


def evaluate_json_with_ngrams(
    file_path: str,
    threshold: float = 0.5,
) -> None:
    if not os.path.exists(file_path):
        print(f"[Error] JSON file not found at: '{file_path}'")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)

    data_entries = json_obj.get("data", [])
    if not data_entries:
        print("[Error] No entries found in the JSON 'data' field.")
        return

    llm_name = json_obj.get("llm_name", "unknown_model")
    task_name = json_obj.get("task_name", "question_answering")
    relation_name = str(json_obj.get("relation_name", "N/A"))

    passed_count = 0
    records: List[Dict] = []

    for idx, item in enumerate(data_entries):
        shot_id = item.get("id", idx)
        src_text, fol_text = extract_texts(item)
        if not src_text or not fol_text:
            continue

        # Calculate N-gram metrics (1-gram, 2-gram, 3-gram)
        ngram_scores = calculate_ngram_metrics(src_text, fol_text)
        
        # Example Pass condition: using average or specific n-gram score against threshold
        avg_score = sum(ngram_scores.values()) / len(ngram_scores)
        is_pass = avg_score >= threshold

        if is_pass:
            passed_count += 1

        records.append({
            "shot_id": shot_id,
            "status": "PASS" if is_pass else "FAIL",
            "1_gram": ngram_scores["1_gram"],
            "2_gram": ngram_scores["2_gram"],
            "3_gram": ngram_scores["3_gram"],
            "avg_ngram_score": round(avg_score, 4),
            "source_text": src_text,
            "transformed_text": fol_text,
        })

    pass_percentage = (passed_count / len(records) * 100.0) if records else 0.0

    print("\n================ N-GRAM EVALUATION RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Threshold (Avg N-gram):   {threshold}")
    print(f"Total Evaluated Shots:    {len(records)}")
    print(f"Passed Threshold:         {passed_count}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("===========================================================")

    output_dir = os.path.join("logs", "Ngram", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Export detailed records to CSV
    csv_filename = f"{llm_name}_rel{relation_name}_{date_str}_ngram.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Shot_ID", "Status", "1_gram", "2_gram", "3_gram", "Avg_Score", "Source_Text", "Transformed_Text"])
        for r in records:
            writer.writerow([
                f"#{r['shot_id']}",
                r["status"],
                f"{r['1_gram']:.4f}",
                f"{r['2_gram']:.4f}",
                f"{r['3_gram']:.4f}",
                f"{r['avg_ngram_score']:.4f}",
                r["source_text"],
                r["transformed_text"],
            ])
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    # 2. Export summary log to TXT
    log_filename = f"{llm_name}_rel{relation_name}_{date_str}_ngram_summary.txt"
    log_path = os.path.join(output_dir, log_filename)

    summary_lines = [
        "=" * 60,
        "             N-GRAM EVALUATION SUMMARY LOG",
        "=" * 60,
        f"Timestamp:                 {date_str}",
        f"Model Evaluated:           {llm_name}",
        f"Task Name:                 {task_name}",
        f"Relation ID:               {relation_name}",
        f"Score Threshold:           {threshold}",
        "-" * 60,
        f"Total Evaluated Shots:     {len(records):,}",
        f"Passed Threshold:          {passed_count:,}",
        f"Pass Percentage:           {pass_percentage:.2f}%",
        "=" * 60,
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Saved] Summary log saved to: '{log_path}'")

    # 3. Export complete results to JSON
    json_filename = f"{llm_name}_rel{relation_name}_{date_str}_ngram.json"
    json_path = os.path.join(output_dir, json_filename)

    json_output = {
        "timestamp": date_str,
        "model_evaluated": llm_name,
        "task_name": task_name,
        "relation_name": relation_name,
        "threshold": threshold,
        "total_shots": len(records),
        "passed_threshold": passed_count,
        "pass_percentage": round(pass_percentage, 2),
        "records": records,
    }

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
    print(f"[Saved] JSON log saved to: '{json_path}'")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate JSON dataset using N-gram metrics (1, 2, 3-gram).")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the evaluated test result JSON file.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Threshold for passing (default: 0.5).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_json_with_ngrams(file_path=args.path, threshold=args.threshold)
