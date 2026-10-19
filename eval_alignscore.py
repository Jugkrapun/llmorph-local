import argparse
import csv
import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

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
    """
    Pairs source and followup n-grams using SequenceMatcher.
    Explicitly handles 'equal', 'replace', 'delete', and 'insert' opcodes.
    Deleted and inserted words are mapped to empty strings to ensure proper penalization.
    """
    if not src_ngrams or not fol_ngrams:
        return []

    matcher = SequenceMatcher(None, src_ngrams, fol_ngrams)
    pairs = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for s_idx, f_idx in zip(range(i1, i2), range(j1, j2)):
                pairs.append((src_ngrams[s_idx], fol_ngrams[f_idx]))
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
        return "Synonym", 0.7
    elif rel_str == "10":
        return "Antonym", 0.3
    else:
        print(f"[Warning] Unknown relation_name '{rel_str}', defaulting to Synonym (>= 0.7)")
        return "Synonym", 0.7


def evaluate_chunks_alignscore(
    file_path: str,
    ckpt_path: str = "AlignScore-base.ckpt",
    model_name: str = "roberta-base",
    batch_size: int = 64,
    ngram_order: int = 1,
) -> None:
    """Extracts all changed/inserted/deleted n-gram chunks across the dataset, evaluates them, and computes pass percentage."""
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

    all_extracted_chunks = []

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
            if s_chunk.lower() != f_chunk.lower():
                all_extracted_chunks.append({
                    "shot_id": shot_id,
                    "source_chunk": s_chunk,
                    "transformed_chunk": f_chunk
                })

    if not all_extracted_chunks:
        print("[Error] No modified n-gram chunks found for evaluation.")
        return

    print(f"\n[Info] Found a total of {len(all_extracted_chunks):,} evaluated chunks (diff/insert/delete).")

    unique_pairs = list(set((c["source_chunk"], c["transformed_chunk"]) for c in all_extracted_chunks))
    print(f"[Init] Deduplicated to {len(unique_pairs):,} unique chunk pairs for batch inference.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Loading AlignScore ({model_name}) on {device}...")

    scorer = AlignScore(
        model=model_name,
        batch_size=batch_size,
        device=device,
        ckpt_path=ckpt_path,
        evaluation_mode="nli_sp",
    )

    print(f"[Run] Computing AlignScore for unique chunk pairs...")
    batch_src = [p[0] for p in unique_pairs]
    batch_fol = [p[1] for p in unique_pairs]
    raw_scores = scorer.score(contexts=batch_src, claims=batch_fol)

    pair_score_lookup = {pair: float(sc) for pair, sc in zip(unique_pairs, raw_scores)}

    passed_chunks_count = 0
    records = []

    for item in all_extracted_chunks:
        s_chunk = item["source_chunk"]
        f_chunk = item["transformed_chunk"]
        score_val = pair_score_lookup.get((s_chunk, f_chunk), 0.0)

        if mode == "Synonym":
            is_pass = score_val >= align_threshold
        else:
            is_pass = score_val <= align_threshold

        if is_pass:
            passed_chunks_count += 1

        records.append({
            "shot_id": item["shot_id"],
            "status": "PASS" if is_pass else "FAIL",
            "align_score": round(score_val, 4),
            "source_chunk": s_chunk,
            "transformed_chunk": f_chunk
        })

    total_chunks = len(records)
    pass_percentage = (passed_chunks_count / total_chunks * 100.0) if total_chunks > 0 else 0.0

    print("\n================ CHUNK-LEVEL ALIGNSCORE RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Mode:                     {mode} (Threshold: {align_threshold})")
    print(f"N-gram Order Evaluated:   {ngram_order}-gram")
    print(f"Total Evaluated Chunks:   {total_chunks:,}")
    print(f"Passed Chunks:            {passed_chunks_count:,}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("================================================================")

    # --- Dynamic output directory derived from file path (-p) ---
    path_dir_name = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
    if not path_dir_name or path_dir_name in ["results", "data", "."]:
        path_dir_name = os.path.splitext(os.path.basename(file_path))[0]

    output_dir = os.path.join("logs", path_dir_name, "alignscore_chunks")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    csv_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Shot_ID", "Status", "AlignScore", "Source_Chunk", "Transformed_Chunk"])
        for r in records:
            writer.writerow([
                f"#{r['shot_id']}",
                r["status"],
                f"{r['align_score']:.4f}",
                r["source_chunk"],
                r["transformed_chunk"],
            ])
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    log_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram_summary.txt"
    log_path = os.path.join(output_dir, log_filename)

    summary_lines = [
        "=" * 60,
        f"        CHUNK-LEVEL {ngram_order}-GRAM ALIGNSCORE SUMMARY",
        "=" * 60,
        f"Timestamp:                 {date_str}",
        f"Model Evaluated:           {llm_name}",
        f"Task Name:                 {task_name}",
        f"Relation ID:               {relation_name}",
        f"Evaluation Mode:           {mode}",
        f"Score Threshold:           {align_threshold}",
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

    json_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram.json"
    json_path = os.path.join(output_dir, json_filename)

    json_output = {
        "timestamp": date_str,
        "model_evaluated": llm_name,
        "task_name": task_name,
        "relation_name": relation_name,
        "evaluation_mode": mode,
        "score_threshold": align_threshold,
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
    parser = argparse.ArgumentParser(description="Evaluate all changed/inserted/deleted n-gram chunks using SequenceMatcher.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the evaluated test result JSON file.")
    parser.add_argument("-c", "--ckpt", type=str, default="AlignScore-base.ckpt", help="Path to checkpoint (.ckpt).")
    parser.add_argument("-m", "--model", type=str, default="roberta-base", help="Backbone model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("-g", "--ngram", type=int, default=1, help="Specify n-gram order (e.g., -g 1, -g 2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_chunks_alignscore(
        file_path=args.path,
        ckpt_path=args.ckpt,
        model_name=args.model,
        batch_size=args.batch_size,
        ngram_order=args.ngram,
    )
