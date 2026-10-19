import argparse
import csv
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import nltk
import spacy
import torch
from transformers import AutoConfig, AutoTokenizer
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

    try:
        self.config = self.model.config
    except AttributeError:
        self.config = AutoConfig.from_pretrained(model)

    try:
        self.tokenizer = self.model.tokenizer
    except AttributeError:
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    self.spacy = spacy.load("en_core_web_sm")
    self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    self.softmax = torch.nn.Softmax(dim=-1)
    self.smart_type = "smart-n"
    self.smart_n_metric = "f1"
    self.disable_progress_bar_in_inference = True
    self.verbose = False


align_inf.Inferencer.__init__ = _patched_inferencer_init
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Tokenizes input text into a list of words."""
    return re.findall(r"\w+", text)


def get_ngram_strings(tokens: List[str], n: int) -> List[str]:
    """Generates sliding-window n-gram strings of size n from a token list."""
    if not tokens:
        return []
    if len(tokens) < n:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def pair_ngrams(src_ngrams: List[str], fol_ngrams: List[str]) -> List[Tuple[str, str]]:
    """Pairs source n-grams and followup n-grams positionally (proportional alignment if lengths differ)."""
    if not src_ngrams or not fol_ngrams:
        return []
    if len(src_ngrams) == len(fol_ngrams):
        return list(zip(src_ngrams, fol_ngrams))

    pairs = []
    max_len = max(len(src_ngrams), len(fol_ngrams))
    for i in range(max_len):
        s_idx = min(int(i * len(src_ngrams) / max_len), len(src_ngrams) - 1)
        f_idx = min(int(i * len(fol_ngrams) / max_len), len(fol_ngrams) - 1)
        pairs.append((src_ngrams[s_idx], fol_ngrams[f_idx]))
    return pairs


def compute_ngram_alignscores_for_order(
    scorer: AlignScore,
    contexts: List[str],
    claims: List[str],
    n: int,
    eval_only_changed: bool = False,
) -> Tuple[List[float], List[int]]:
    """
    Segments source and followup texts into n-grams first, then evaluates each n-gram pair using AlignScore.
    Identical n-gram pairs automatically receive 1.0 to avoid redundant GPU inference.
    """
    unique_changed_pairs = set()
    shot_pair_lists = []

    for src, fol in zip(contexts, claims):
        src_tokens = tokenize(src)
        fol_tokens = tokenize(fol)
        src_ng = get_ngram_strings(src_tokens, n)
        fol_ng = get_ngram_strings(fol_tokens, n)

        pairs = pair_ngrams(src_ng, fol_ng)
        shot_pair_lists.append(pairs)

        for s_chunk, f_chunk in pairs:
            if s_chunk.lower() != f_chunk.lower():
                unique_changed_pairs.add((s_chunk, f_chunk))

    # Batch evaluate all unique modified n-gram pairs with AlignScore
    pair_score_lookup: Dict[Tuple[str, str], float] = {}
    if unique_changed_pairs:
        pair_list = list(unique_changed_pairs)
        batch_src = [p[0] for p in pair_list]
        batch_fol = [p[1] for p in pair_list]
        print(f"[Run] Evaluating {len(pair_list):,} unique modified {n}-gram pairs with AlignScore...")
        raw_scores = scorer.score(contexts=batch_src, claims=batch_fol)
        for pair, sc in zip(pair_list, raw_scores):
            pair_score_lookup[pair] = float(sc)

    shot_avg_scores = []
    shot_changed_counts = []

    for pairs in shot_pair_lists:
        if not pairs:
            shot_avg_scores.append(0.0)
            shot_changed_counts.append(0)
            continue

        scores = []
        changed_cnt = 0
        for s_chunk, f_chunk in pairs:
            if s_chunk.lower() == f_chunk.lower():
                if not eval_only_changed:
                    scores.append(1.0)
            else:
                changed_cnt += 1
                scores.append(pair_score_lookup.get((s_chunk, f_chunk), 0.0))

        if not scores:
            scores = [1.0]

        shot_avg_scores.append(sum(scores) / len(scores))
        shot_changed_counts.append(changed_cnt)

    return shot_avg_scores, shot_changed_counts


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


def evaluate_json_with_ngram_alignscore(
    file_path: str,
    ckpt_path: str = "AlignScore-base.ckpt",
    model_name: str = "roberta-base",
    batch_size: int = 64,
    ngram_order: int = 1,
    diff_only: bool = False,
) -> None:
    """Main function to segment texts into a specific n-gram order and evaluate pairs using AlignScore."""
    if not os.path.exists(file_path):
        print(f"[Error] JSON file not found at: '{file_path}'")
        return

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

    scorer = AlignScore(
        model=model_name,
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
    )

    print(f"\n[Step] Segmenting texts into {ngram_order}-gram and computing AlignScore...")
    scores_n, changed_n = compute_ngram_alignscores_for_order(
        scorer=scorer,
        contexts=contexts,
        claims=claims,
        n=ngram_order,
        eval_only_changed=diff_only,
    )

    passed_count = 0
    records: List[Dict] = []

    for i, (shot_id, src, fol) in enumerate(zip(shot_ids, contexts, claims)):
        score_val = scores_n[i]

        if mode == "Synonym":
            is_pass = score_val >= align_threshold
        else:
            is_pass = score_val <= align_threshold

        if is_pass:
            passed_count += 1

        record_entry = {
            "shot_id": shot_id,
            "status": "PASS" if is_pass else "FAIL",
            f"{ngram_order}_gram_alignscore": round(score_val, 4),
            f"{ngram_order}_gram_changed_chunks": changed_n[i],
            "source_text": src,
            "transformed_text": fol,
        }
        records.append(record_entry)

    pass_percentage = (passed_count / len(records) * 100.0) if records else 0.0

    print("\n================ N-GRAM SEGMENTED ALIGNSCORE RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Mode:                     {mode} (Threshold: {align_threshold})")
    print(f"N-gram Order Evaluated:   {ngram_order}-gram")
    print(f"Scoring Scope:            {'Modified N-grams Only' if diff_only else 'All N-grams (Full Text)'}")
    print(f"Total Evaluated Shots:    {len(records)}")
    print(f"Passed Threshold:         {passed_count}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("=====================================================================")

    output_dir = os.path.join("logs", mode.capitalize(), "alignscore_ngram")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 1. Export CSV
    csv_headers = ["Shot_ID", "Status", f"{ngram_order}_gram_AlignScore", f"{ngram_order}_gram_Changed_Chunks", "Source_Text", "Transformed_Text"]
    csv_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram_alignscore.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(csv_headers)
        for r in records:
            writer.writerow([
                f"#{r['shot_id']}",
                r["status"],
                f"{r[f'{ngram_order}_gram_alignscore']:.4f}",
                r[f"{ngram_order}_gram_changed_chunks"],
                r["source_text"],
                r["transformed_text"],
            ])
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    # 2. Export TXT Summary
    log_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram_alignscore_summary.txt"
    log_path = os.path.join(output_dir, log_filename)

    summary_lines = [
        "=" * 60,
        f"       {ngram_order}-GRAM SEGMENTED ALIGNSCORE SUMMARY",
        "=" * 60,
        f"Timestamp:                 {date_str}",
        f"Model Evaluated:           {llm_name}",
        f"Task Name:                 {task_name}",
        f"Relation ID:               {relation_name}",
        f"Evaluation Mode:           {mode}",
        f"Score Threshold:           {align_threshold}",
        f"N-gram Order:              {ngram_order}",
        f"Scoring Scope:             {'Modified N-grams Only' if diff_only else 'All N-grams (Full Text)'}",
        "-" * 60,
        f"Total Evaluated Shots:     {len(records):,}",
        f"Passed Threshold:          {passed_count:,}",
        f"Pass Percentage:           {pass_percentage:.2f}%",
        "=" * 60,
    ]

    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Saved] Summary log saved to: '{log_path}'")

    # 3. Export JSON
    json_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram_alignscore.json"
    json_path = os.path.join(output_dir, json_filename)

    json_output = {
        "timestamp": date_str,
        "model_evaluated": llm_name,
        "task_name": task_name,
        "relation_name": relation_name,
        "evaluation_mode": mode,
        "score_threshold": align_threshold,
        "ngram_order": ngram_order,
        "diff_only": diff_only,
        "total_shots": len(records),
        "passed_threshold": passed_count,
        "pass_percentage": round(pass_percentage, 2),
        "records": records,
    }

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
    print(f"[Saved] JSON log saved to: '{json_path}'")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text pairs using a single specified N-gram order with AlignScore.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the evaluated test result JSON file.")
    parser.add_argument("-c", "--ckpt", type=str, default="AlignScore-base.ckpt", help="Path to checkpoint (.ckpt).")
    parser.add_argument("-m", "--model", type=str, default="roberta-base", help="Backbone model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("-g", "--ngram", type=int, default=1, help="Specify a single n-gram order to evaluate (e.g., -g 1, -g 2, or -g 3).")
    parser.add_argument("--diff-only", action="store_true", help="If set, computes score ONLY over modified n-gram chunks.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_json_with_ngram_alignscore(
        file_path=args.path,
        ckpt_path=args.ckpt,
        model_name=args.model,
        batch_size=args.batch_size,
        ngram_order=args.ngram,
        diff_only=args.diff_only,
    )
