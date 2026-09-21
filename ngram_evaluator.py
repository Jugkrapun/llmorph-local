import csv
import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# Headless backend to prevent X11 display pixmap overflow errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from sentence_transformers import SentenceTransformer, util


class Tokenizer:
    """Wraps spaCy for tokenization and per-shot token-length scanning."""

    def __init__(self, spacy_model: str = "en_core_web_md"):
        # Load spaCy pipeline strictly for tokenization
        self.nlp = spacy.load(spacy_model, disable=["parser", "ner", "tagger"])

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text using spaCy, stripping standalone punctuation and whitespace."""
        doc = self.nlp(text)
        return [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]

    @staticmethod
    def _extract_texts(item: dict) -> Tuple[str, Optional[str]]:
        """Pulls (source_text, followup_text) out of one dataset entry."""
        src_input = item.get("source_input", [])
        src_text = " ".join(src_input) if isinstance(src_input, list) else str(src_input)

        followup_list = item.get("followup_inputs", [])
        if not followup_list or not followup_list[0]:
            return src_text, None

        fol_target = followup_list[0]
        fol_text = " ".join(fol_target) if isinstance(fol_target, list) else str(fol_target)
        return src_text, fol_text

    def scan_token_lengths(self, data_entries: List[dict]) -> Dict[int, bool]:
        """Scans dataset to determine equal length vs mismatched shots."""
        equality_map: Dict[int, bool] = {}
        mismatched_count = 0

        print("\n--- Running Dataset Token Length Scan ---")
        for shot_idx, item in enumerate(data_entries):
            shot_id = item.get("id", shot_idx)
            src_text, fol_text = self._extract_texts(item)
            if fol_text is None:
                continue

            len_src = len(self.tokenize(src_text))
            len_fol = len(self.tokenize(fol_text))

            if len_src == len_fol:
                equality_map[shot_id] = True
            else:
                equality_map[shot_id] = False
                mismatched_count += 1
                print(
                    f"  - Shot #{shot_id}: Mismatch (Source: {len_src} tokens | "
                    f"Followup: {len_fol} tokens | Diff: {len_fol - len_src:+d}) "
                    f"-> routed to Anchor Alignment"
                )

        print(
            f"\n[Scan Summary] Equal length shots: {len(data_entries) - mismatched_count} "
            f"| Mismatched shots: {mismatched_count}"
        )
        return equality_map


class NgramExtractor:
    """Builds position-aligned n-gram pairs from two token sequences."""

    @staticmethod
    def extract_standard_sliding_ngrams(
        tokens1: List[str], tokens2: List[str], n: int
    ) -> List[Tuple[int, str, str]]:
        pairs = []
        if n == 1:
            ngrams1 = [[t] for t in tokens1]
            ngrams2 = [[t] for t in tokens2]
        else:
            ngrams1 = [tokens1[i : i + n] for i in range(len(tokens1) - n + 1)] if len(tokens1) >= n else [tokens1]
            ngrams2 = [tokens2[i : i + n] for i in range(len(tokens2) - n + 1)] if len(tokens2) >= n else [tokens2]

        min_len = min(len(ngrams1), len(ngrams2))
        for pos_idx in range(min_len):
            span1 = " ".join(ngrams1[pos_idx]).strip()
            span2 = " ".join(ngrams2[pos_idx]).strip()

            if span1 and span2 and span1 != span2:
                pairs.append((pos_idx, span1, span2))

        return pairs

    @staticmethod
    def _build_alignment_map(tokens1: List[str], tokens2: List[str]) -> List[int]:
        matcher = SequenceMatcher(None, tokens1, tokens2)
        map_j: List[Optional[int]] = [None] * len(tokens1)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    map_j[i1 + k] = j1 + k
            elif tag == "replace":
                len1, len2 = i2 - i1, j2 - j1
                for k in range(len1):
                    map_j[i1 + k] = j1 + min(k, max(len2 - 1, 0))
            elif tag == "delete":
                for k in range(i2 - i1):
                    map_j[i1 + k] = min(j1, max(len(tokens2) - 1, 0))

        for i in range(len(map_j)):
            if map_j[i] is None:
                map_j[i] = min(i, max(len(tokens2) - 1, 0))

        return map_j  # type: ignore[return-value]

    def extract_anchor_aligned_ngrams(
        self, tokens1: List[str], tokens2: List[str], n: int
    ) -> List[Tuple[int, str, str]]:
        if not tokens1:
            return []

        map_j = self._build_alignment_map(tokens1, tokens2)
        len1 = len(tokens1)
        window1 = (
            [tokens1[i : i + n] for i in range(len1 - n + 1)]
            if len1 >= n
            else [tokens1]
        )

        pairs = []
        for pos_idx, w1 in enumerate(window1):
            j_start = map_j[pos_idx] if pos_idx < len(map_j) else None
            if j_start is None or not tokens2:
                continue
            j_start = max(0, min(j_start, len(tokens2) - 1))
            w2 = tokens2[j_start : j_start + n]
            if len(w2) < n:
                w2 = tokens2[max(0, len(tokens2) - n):]

            span1 = " ".join(w1).strip()
            span2 = " ".join(w2).strip()
            if span1 and span2 and span1 != span2:
                pairs.append((pos_idx, span1, span2))

        return pairs


class SimilarityScorer:
    """
    Computes Token-wise Cosine Similarity using SentenceTransformer exclusively
    across all N-Gram levels.
    """

    def __init__(self, sentence_model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(sentence_model_name)

    def _get_single_word_similarity(self, w1: str, w2: str) -> float:
        """Computes similarity between two single words using SentenceTransformer embeddings."""
        if w1 == w2:
            return 1.0

        emb1 = self.model.encode(w1, convert_to_tensor=True)
        emb2 = self.model.encode(w2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).item())

    def calculate_token_averaged_similarity(
        self, orig_text: str, mod_text: str, gram_level: int, mode: str
    ) -> float:
        """
        Calculates similarity per token position with SentenceTransformer,
        then computes the average score (e.g., divided by 2 for 2-gram, by 3 for 3-gram).
        """
        words1 = orig_text.split()
        words2 = mod_text.split()

        # For 1-Gram (Direct single word)
        if gram_level == 1:
            raw_score = self._get_single_word_similarity(orig_text, mod_text)
            return -raw_score if mode == "Antonym" else raw_score

        # For 2-Gram, 3-Gram, etc.
        token_scores = []
        max_tokens = max(len(words1), len(words2), gram_level)

        for i in range(max_tokens):
            w1 = words1[i] if i < len(words1) else ""
            w2 = words2[i] if i < len(words2) else ""

            if w1 and w2:
                sim = self._get_single_word_similarity(w1, w2)
            else:
                sim = 0.0

            token_scores.append(sim)

        # Average across N-gram elements: (sim1 + sim2 + ... + simN) / N
        avg_score = sum(token_scores) / len(token_scores) if token_scores else 0.0

        return -avg_score if mode == "Antonym" else avg_score


class DatasetEvaluator:
    """Runs extraction + scoring across an entire dataset for one gram level."""

    def __init__(self, tokenizer: Tokenizer, extractor: NgramExtractor, scorer: SimilarityScorer):
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.scorer = scorer

    def evaluate(
        self,
        data_entries: List[dict],
        equality_map: Dict[int, bool],
        gram_level: int,
        mode: str,
        threshold: float = 0.8,
    ) -> Dict[str, Any]:
        all_pairs_collected = []
        total_passed = 0
        total_transformations = 0

        print(f"\nProcessing {len(data_entries)} shots ({gram_level}-Gram Token-wise Average)...")

        for shot_idx, item in enumerate(data_entries):
            shot_id = item.get("id", shot_idx)
            src_text, fol_text = self.tokenizer._extract_texts(item)
            if fol_text is None:
                continue

            tokens_src = self.tokenizer.tokenize(src_text)
            tokens_fol = self.tokenizer.tokenize(fol_text)

            is_equal = equality_map.get(shot_id, True)
            if is_equal:
                pairs = self.extractor.extract_standard_sliding_ngrams(tokens_src, tokens_fol, gram_level)
                method_used = "Standard Sliding"
            else:
                pairs = self.extractor.extract_anchor_aligned_ngrams(tokens_src, tokens_fol, gram_level)
                method_used = "Anchor Alignment"

            for pos_idx, orig_text, mod_text in pairs:
                sim_score = self.scorer.calculate_token_averaged_similarity(
                    orig_text, mod_text, gram_level, mode
                )

                is_valid = (sim_score >= threshold) if mode == "Synonym" else (sim_score <= threshold)

                if is_valid:
                    total_passed += 1
                total_transformations += 1

                all_pairs_collected.append({
                    "shot_id": shot_id,
                    "word_pos": pos_idx,
                    "method": method_used,
                    "original": orig_text,
                    "transformed": mod_text,
                    "cosine_similarity": round(sim_score, 4),
                    "pass_threshold": is_valid,
                })

        micro_precision = (total_passed / total_transformations) if total_transformations > 0 else 0.0

        return {
            "total_shots": len(data_entries),
            "total_transformations": total_transformations,
            "total_passed": total_passed,
            "micro_precision": micro_precision,
            "metrics": all_pairs_collected,
        }


class ResultExporter:
    """Writes evaluation results to CSV and a summary plot image."""

    @staticmethod
    def export_csv(
        metric_list: List[dict],
        mode: str,
        output_dir: str,
        llm_name: str,
        relation_name: str,
        gram_level: int,
        date_time_str: str,
    ) -> None:
        metric_list.sort(key=lambda x: (int(x["shot_id"]), int(x["word_pos"])))

        csv_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_time_str}_{gram_level}gram_sequential.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                "Shot_ID",
                "Position",
                "Status",
                "Original_NGram",
                "Transformed_NGram",
                "Cosine_Similarity",
            ])
            for item in metric_list:
                status_text = "PASS" if item["pass_threshold"] else "FAIL"
                writer.writerow([
                    f"#{item['shot_id']}",
                    item["word_pos"],
                    status_text,
                    item["original"],
                    item["transformed"],
                    f"{item['cosine_similarity']:.4f}",
                ])

        print(f"[Saved] Evaluation CSV correctly formatted and saved to: '{csv_path}'")

    @staticmethod
    def save_summary_plot(
        summary_res: dict,
        mode: str,
        gram_level: int,
        llm_name: str,
        task_name: str,
        relation_name: str,
        output_dir: str,
        date_str: str,
    ) -> None:
        headers = ["Metric Description", "Value"]
        table_data = [
            ["Model Evaluated", llm_name],
            ["Relation ID", str(relation_name)],
            ["Evaluation Level", f"{gram_level}-Gram"],
            ["Cosine Engine", "SentenceTransformer (all-mpnet-base-v2)"],
            ["Total Evaluated Shots", f"{summary_res['total_shots']:,}"],
            ["Total Evaluated N-Grams", f"{summary_res['total_transformations']:,}"],
            ["Passed Transformations", f"{summary_res['total_passed']:,}"],
            ["Overall Dataset Precision", f"{summary_res['micro_precision'] * 100:.2f}%"],
        ]

        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        ax.axis("off")

        plt.title(
            f"Embedding Evaluation Summary ({summary_res['total_shots']} Shots)\n"
            f"Task: {task_name} | Mode: {mode} | Level: {gram_level}-Gram",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        table = ax.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2B4C7E")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#F8F9FA" if row % 2 == 0 else "#FFFFFF")

        plot_filename = f"{llm_name}_rel{relation_name}_{mode.lower()}_{date_str}_{gram_level}gram_summary.png"
        plot_path = os.path.join(output_dir, plot_filename)

        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] Summary image saved to: '{plot_path}'")


class EvaluationPipeline:
    """Orchestrates the full CLI flow."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_md",
        sentence_model_name: str = "all-mpnet-base-v2",
    ):
        self.tokenizer = Tokenizer(spacy_model)
        self.extractor = NgramExtractor()
        self.scorer = SimilarityScorer(sentence_model_name)
        self.evaluator = DatasetEvaluator(self.tokenizer, self.extractor, self.scorer)
        self.exporter = ResultExporter()

    @staticmethod
    def _prompt_gram_levels() -> List[int]:
        gram_input = input("\nEnter N-Gram levels to run, comma-separated (e.g., 1,2,3): ").strip()
        try:
            gram_levels = sorted(set(int(g.strip()) for g in gram_input.split(",") if g.strip()))
            return [g for g in gram_levels if g >= 1] or [1, 2, 3]
        except ValueError:
            return [1, 2, 3]

    @staticmethod
    def _prompt_mode() -> Tuple[str, float]:
        print("\nSelect Evaluation Mode:")
        print("1: Synonym (Cosine >= 0.8)")
        print("2: Antonym (Cosine <= -0.8)")
        mode_choice = input("Enter choice (1 or 2): ").strip()
        if mode_choice == "2":
            return "Antonym", -0.8
        return "Synonym", 0.8

    def run(self) -> None:
        file_path = input("Enter JSON file path: ").strip()
        if not os.path.exists(file_path):
            print(f"[Error] File not found: '{file_path}'")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)

        data_entries = json_obj.get("data", [])
        if not data_entries:
            print("[Error] No entries found in JSON 'data' field.")
            return

        equality_map = self.tokenizer.scan_token_lengths(data_entries)

        gram_levels = self._prompt_gram_levels()
        mode, threshold = self._prompt_mode()

        llm_name = json_obj.get("llm_name", "unknown_model")
        task_name = json_obj.get("task_name", "question_answering")
        relation_name = json_obj.get("relation_name", "N/A")
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for gram_level in gram_levels:
            results = self.evaluator.evaluate(data_entries, equality_map, gram_level, mode, threshold)

            print(f"\n================ DATASET EVALUATION RESULTS ({gram_level}-GRAM) ================")
            print(f"Model:                 {llm_name}")
            print(f"Task:                  {task_name} (Relation {relation_name})")
            print(f"Cosine Engine:         SentenceTransformer (all-mpnet-base-v2)")
            print(f"Threshold:             {threshold}")
            print(f"Total Shots:           {results['total_shots']}")
            print(f"Total N-grams Found:   {results['total_transformations']}")
            print(f"Total Passed Criteria: {results['total_passed']}")
            print(f"Overall Precision:     {results['micro_precision'] * 100:.2f}%")
            print("============================================================")

            output_dir = os.path.join("plots", mode.capitalize(), f"plot-{gram_level}gram")
            os.makedirs(output_dir, exist_ok=True)

            if results["metrics"]:
                self.exporter.export_csv(
                    results["metrics"], mode, output_dir, llm_name, relation_name, gram_level, date_str
                )

            self.exporter.save_summary_plot(
                results, mode, gram_level, llm_name, task_name, relation_name, output_dir, date_str
            )


if __name__ == "__main__":
    EvaluationPipeline().run()
