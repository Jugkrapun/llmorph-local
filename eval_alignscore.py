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
from nltk.corpus import wordnet
from transformers import AutoConfig, AutoTokenizer
from alignscore import AlignScore
import alignscore.inference as align_inf
from alignscore.model import BERTAlignModel


# ==============================================================================
# Monkey-patch AlignScore Inferencer to support PyTorch Lightning 2.x
# ==============================================================================
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
# ==============================================================================


def tokenize(text: str) -> List[str]:
    """Tokenizes input text into a list of word tokens using regex word boundaries."""
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


def resolve_evaluation_config(relation_name: str) -> Tuple[str, float]:
    """Automatically infers evaluation mode and threshold based on relation ID."""
    rel_str = str(relation_name).strip()
    if rel_str == "8":
        return "Synonym", 0.8
    elif rel_str == "10":
        return "Antonym", 0.8
    else:
        print(f"[Warning] Unknown relation_name '{rel_str}', defaulting to Synonym (Threshold >= 0.8)")
        return "Synonym", 0.8


def evaluate_dataset(
    file_path: str,
    ckpt_path: str = "AlignScore-base.ckpt",
    model_name: str = "roberta-base",
    batch_size: int = 64,
    ngram_order: int = 1,
) -> None:
    """
    Comprehensive Unified Evaluation Algorithm:
    - Relation 8 (Synonym): Evaluates Source Chunk directly vs Transformed Chunk (Threshold >= 0.7).
    - Relation 10 (Antonym): Evaluates WordNet Antonym vs Transformed Chunk (Threshold >= 0.5).
    """
    if not os.path.exists(file_path):
        print(f"[Error] JSON file not found at: '{file_path}'")
        return

    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
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
    
    # Automatically resolve evaluation mode and threshold based on relation ID
    mode, align_threshold = resolve_evaluation_config(relation_name)

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
            if not s_chunk.strip() and not f_chunk.strip():
                continue
            if s_chunk.lower() == f_chunk.lower():
                continue

            if mode == "Synonym":
                # Synonym Case: Compare source chunk directly with transformed output chunk
                all_evaluated_items.append({
                    "shot_id": shot_id,
                    "context_chunk": s_chunk,
                    "claim_chunk": f_chunk,
                    "display_source": s_chunk,
                    "display_target": f_chunk
                })
            else:
                # Antonym Case: Compare true WordNet antonym with transformed output chunk
                wordnet_ant = get_wordnet_antonym(s_chunk)
                if not wordnet_ant:
                    continue
                all_evaluated_items.append({
                    "shot_id": shot_id,
                    "context_chunk": wordnet_ant,
                    "claim_chunk": f_chunk,
                    "display_source": s_chunk,
                    "wordnet_antonym": wordnet_ant,
                    "display_target": f_chunk
                })

    if not all_evaluated_items:
        print("[Error] No valid chunk pairs found for evaluation.")
        return

    print(f"\n[Info] Found a total of {len(all_evaluated_items):,} chunks to evaluate under mode: {mode}.")

    # Deduplicate unique pairs for efficient batch inference
    unique_pairs = list(set((item["context_chunk"], item["claim_chunk"]) for item in all_evaluated_items))
    print(f"[Init] Deduplicated to {len(unique_pairs):,} unique pairs for batch inference.")

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
    
    pair_score_lookup = {}
    valid_pairs = []

    for pair in unique_pairs:
        ctx_c, clm_c = pair
        if not ctx_c.strip() or not clm_c.strip():
            pair_score_lookup[pair] = 0.0
        else:
            valid_pairs.append(pair)

    if valid_pairs:
        batch_ctx = [p[0] for p in valid_pairs]
        batch_clm = [p[1] for p in valid_pairs]
        raw_scores = scorer.score(contexts=batch_ctx, claims=batch_clm)
        for pair, sc in zip(valid_pairs, raw_scores):
            pair_score_lookup[pair] = float(sc)

    passed_count = 0
    records = []

    for item in all_evaluated_items:
        ctx_c = item["context_chunk"]
        clm_c = item["claim_chunk"]
        score_val = pair_score_lookup.get((ctx_c, clm_c), 0.0)

        # PASS if score meets or exceeds threshold
        is_pass = score_val >= align_threshold
        if is_pass:
            passed_count += 1

        record_entry = {
            "shot_id": item["shot_id"],
            "status": "PASS" if is_pass else "FAIL",
            "align_score": round(score_val, 4),
            "source_chunk": item["display_source"],
            "transformed_chunk": item["display_target"]
        }
        if mode == "Antonym":
            record_entry["wordnet_antonym"] = item["wordnet_antonym"]

        records.append(record_entry)

    total_chunks = len(records)
    pass_percentage = (passed_count / total_chunks * 100.0) if total_chunks > 0 else 0.0

    print("\n================ COMPREHENSIVE ALIGNSCORE RESULTS ================")
    print(f"Model Evaluated:          {llm_name}")
    print(f"Task:                     {task_name} (Relation {relation_name})")
    print(f"Evaluation Mode:          {mode}")
    print(f"Threshold (>=):           {align_threshold}")
    print(f"N-gram Order Evaluated:   {ngram_order}-gram")
    print(f"Total Evaluated Chunks:   {total_chunks:,}")
    print(f"Passed Chunks:            {passed_count:,}")
    print(f"Pass Percentage:          {pass_percentage:.2f}%")
    print("==================================================================")

    # Automatically structure output directory path: logs/{Mode}/{ProjectName}/alignscore_chunks
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

    output_dir = os.path.join("logs", mode, project_folder, "alignscore")
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_prefix = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{ngram_order}gram"

    # Export CSV file
    csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if mode == "Antonym":
            writer.writerow(["Shot_ID", "Status", "AlignScore", "Source_Chunk", "WordNet_Antonym", "Transformed_Chunk"])
            for r in records:
                writer.writerow([f"#{r['shot_id']}", r["status"], f"{r['align_score']:.4f}", r["source_chunk"], r["wordnet_antonym"], r["transformed_chunk"]])
        else:
            writer.writerow(["Shot_ID", "Status", "AlignScore", "Source_Chunk", "Transformed_Chunk"])
            for r in records:
                writer.writerow([f"#{r['shot_id']}", r["status"], f"{r['align_score']:.4f}", r["source_chunk"], r["transformed_chunk"]])
    print(f"[Saved] CSV results saved to: '{csv_path}'")

    # Export JSON file
    json_path = os.path.join(output_dir, f"{file_prefix}.json")
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump({
            "timestamp": date_str,
            "model_evaluated": llm_name,
            "task_name": task_name,
            "relation_name": relation_name,
            "evaluation_mode": mode,
            "score_threshold": align_threshold,
            "ngram_order": ngram_order,
            "total_chunks": total_chunks,
            "passed_chunks": passed_count,
            "pass_percentage": round(pass_percentage, 2),
            "records": records
        }, f, ensure_ascii=False, indent=4)
    print(f"[Saved] JSON log saved to: '{json_path}'")

    # Export TXT Summary log file
    log_path = os.path.join(output_dir, f"{file_prefix}_summary.txt")
    summary_lines = [
        "=" * 60,
        f"        COMPREHENSIVE ALIGNSCORE SUMMARY ({mode.upper()})",
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
        f"Passed Chunks:             {passed_count:,}",
        f"Pass Percentage:           {pass_percentage:.2f}%",
        "=" * 60,
    ]
    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Saved] Summary log saved to: '{log_path}'")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive evaluation script supporting both Synonym and Antonym modes.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to evaluated test result JSON file.")
    parser.add_argument("-c", "--ckpt", type=str, default="AlignScore-base.ckpt", help="Path to checkpoint (.ckpt).")
    parser.add_argument("-m", "--model", type=str, default="roberta-base", help="Backbone model name.")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("-g", "--ngram", type=int, default=1, help="Specify n-gram order (e.g., -g 1, -g 2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    evaluate_dataset(
        file_path=args.path,
        ckpt_path=args.ckpt,
        model_name=args.model,
        batch_size=args.batch_size,
        ngram_order=args.ngram,
    )
