import asyncio
import json
import aiofiles
import difflib
import re
from typing import Any, Callable, Dict, List, Tuple

from benchmarks.benchmark import BaseBenchmark
from benchmarks.math import MATHBenchmark
from benchmarks.humaneval import HumanEvalBenchmark
from benchmarks.mmlu import MMLUBenchmark
from scripts.logs import logger


class CombinedBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.benchmarks = {
            "math": MATHBenchmark("MATH", "", log_path),
            "humaneval": HumanEvalBenchmark("HumanEval", "", log_path),
            "mmlu": MMLUBenchmark("MMLU", "", log_path)
        }
        self.results = {
            "math": {"scores": [], "costs": []},
            "humaneval": {"scores": [], "costs": []},
            "mmlu": {"scores": [], "costs": []}
        }

    def _format_mmlu_prompt(self, question: str, choices: Any) -> str:
        if isinstance(choices, list) and choices:
            lines = [f"{i}. {choice}" for i, choice in enumerate(choices)]
            return question.strip() + "\n\n" + "\n".join(lines)
        return question.strip()

    def _normalize_text(self, s: str) -> str:
        s = s.strip().lower()
        # Remove punctuation and excessive spaces
        s = re.sub(r"[\s]+", " ", s)
        s = re.sub(r"[^a-z0-9$%.,:;\-_/ ]+", "", s)
        return s

    def _map_prediction_to_choice_letter(self, prediction: str, choices: Any) -> str:
        """Return letter (A, B, ...) inferred from prediction content.
        Attempts in order:
        1) Regex letter extraction (Answer: C, final answer is D, etc.)
        2) Exact/contains match against choice texts (normalized)
        3) Fuzzy match against choice texts
        Returns empty string if no mapping.
        """
        if not isinstance(prediction, str):
            prediction = str(prediction)

        # 0) Numeric option extraction: prefer 0-based, else 1-based
        num_match = re.search(r"\b(\d{1,2})\b", prediction)
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if num_match:
            try:
                raw_num = int(num_match.group(1))
                # First, try as 0-based
                if 0 <= raw_num < len(choices):
                    return labels[raw_num]
                # Then, try as 1-based
                idx = raw_num - 1
                if 0 <= idx < len(choices):
                    return labels[idx]
            except Exception:
                pass

        # 1) Regex letter extraction
        patterns = [
            r"The final answer is\s*:?\s*([A-Z])",
            r"Answer\s*:?\s*([A-Z])",
            r"Option\s*([A-Z])",
            r"Choice\s*([A-Z])",
            r"\b([A-Z])\s*\)",
            r"\b([A-Z])\.",
            r"\b([A-Z])\b",
        ]
        for pat in patterns:
            m = re.search(pat, prediction, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                if letter and letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    return letter

        # No regex letter found; try mapping by content
        if not isinstance(choices, list) or not choices:
            return ""

        norm_pred = self._normalize_text(prediction)

        # 2) Exact/containment match
        normalized_choices = [self._normalize_text(str(c)) for c in choices]
        for idx, norm_choice in enumerate(normalized_choices):
            if not norm_choice:
                continue
            if norm_choice == norm_pred or norm_choice in norm_pred or norm_pred in norm_choice:
                return labels[idx] if idx < len(labels) else ""

        # 3) Fuzzy match
        best_idx = -1
        best_ratio = 0.0
        for idx, norm_choice in enumerate(normalized_choices):
            ratio = difflib.SequenceMatcher(None, norm_choice, norm_pred).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
        if best_idx >= 0 and best_ratio >= 0.85:
            return labels[best_idx] if best_idx < len(labels) else ""

        return ""

    def _unpack_graph_result(self, result: Any) -> Tuple[str, float, Any, Any]:
        prediction = ""
        cost: float = 0.0
        log_prob = None
        selected_operators = None

        if isinstance(result, dict):
            prediction = result.get("prediction") or result.get("answer") or result.get("response") or ""
            try:
                cost = float(result.get("cost", 0.0))
            except Exception:
                cost = 0.0
            log_prob = result.get("log_prob")
            selected_operators = result.get("selected_operators")
        elif isinstance(result, (list, tuple)):
            if len(result) >= 4:
                prediction, cost, log_prob, selected_operators = result[:4]
            elif len(result) == 2:
                prediction, cost = result
            elif len(result) == 1:
                prediction = result[0]
        else:
            prediction = str(result)

        try:
            cost = float(cost)
        except Exception:
            cost = 0.0
        return prediction, cost, log_prob, selected_operators

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        # Load data from the combined dataset file
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, str, int, float]:
        # Support both keys: "benchmark_type" (preferred) and legacy "type"
        benchmark_type = problem.get("benchmark_type") or problem.get("type", "unknown")
        
        if benchmark_type not in self.benchmarks:
            logger.warning(f"Unknown benchmark type: {benchmark_type}")
            return benchmark_type, problem.get("question", ""), "", "", 0, 0.0
        
        benchmark = self.benchmarks[benchmark_type]
        
        # Create a problem dict in the format expected by each benchmark.
        # Be permissive with input schema to support external sources.
        if benchmark_type == "math":
            # Accept either top-level question/answer or math's problem/solution directly
            question = problem.get("question", problem.get("problem", ""))
            answer = problem.get("answer", problem.get("solution", ""))
            formatted_problem = {"problem": question, "solution": answer}
        elif benchmark_type == "humaneval":
            # Map prompt -> question, canonical_solution -> answer if not provided
            prompt = problem.get("question", problem.get("prompt", ""))
            canonical_solution = problem.get("answer", problem.get("canonical_solution", ""))
            formatted_problem = {
                "prompt": prompt,
                "entry_point": problem.get("entry_point", "solution"),
                "test": problem.get("test", ""),
                "canonical_solution": canonical_solution,
            }
        elif benchmark_type == "mmlu":
            # Normalize answer using provided choices and route robustly to graph
            question_text = problem.get("question", "")
            choices = problem.get("choices")
            ans_raw = problem.get("answer", "")

            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if isinstance(choices, list) and choices:
                if isinstance(ans_raw, int):
                    ans = labels[ans_raw] if 0 <= ans_raw < len(labels) else str(ans_raw)
                elif isinstance(ans_raw, str):
                    s = ans_raw.strip()
                    if len(s) == 1 and s.upper() in labels[: len(choices)]:
                        ans = s.upper()
                    else:
                        try:
                            idx = choices.index(ans_raw)
                            ans = labels[idx]
                        except ValueError:
                            ans = s
                else:
                    ans = str(ans_raw)
            else:
                # No explicit choices provided; fallback
                ans = ans_raw
                if isinstance(ans, int):
                    try:
                        ans = "ABCD"[ans]
                    except Exception:
                        ans = str(ans)

            # Try richer call signature first: pass full problem dict
            call_problem = {"question": question_text, "choices": choices, "answer": ans}
            try:
                raw = await graph(call_problem)
            except TypeError:
                # Fallback to prompt string with enumerated options
                prompt_text = self._format_mmlu_prompt(question_text, choices)
                instr = f"\n\nReply with only the index number (0..{max(0, (len(choices) if isinstance(choices, list) else 0) - 1)})."
                raw = await graph(prompt_text + instr)
            except Exception as e:
                try:
                    prompt_text = self._format_mmlu_prompt(question_text, choices)
                    instr = f"\n\nReply with only the index number (0..{max(0, (len(choices) if isinstance(choices, list) else 0) - 1)})."
                    raw = await graph(prompt_text + instr)
                except Exception:
                    logger.error(f"Error evaluating mmlu problem: {e}")
                    return benchmark_type, question_text, str(e), ans, 0, 0.0

            prediction, cost, log_prob, selected_ops = self._unpack_graph_result(raw)

            # Normalize model output to a letter for robust scoring
            pred_letter = self._map_prediction_to_choice_letter(prediction, choices)

            # Score using MMLU benchmark's scorer to avoid changing mmlu.py
            mmlu_bench: MMLUBenchmark = self.benchmarks["mmlu"]
            # Use normalized letter; if empty, fall back to raw prediction
            to_score = pred_letter if pred_letter else prediction
            uni_score, extracted = mmlu_bench.calculate_score(ans, to_score)

            # Log mismatch for debugging when wrong
            if uni_score == 0:
                prompt_text = self._format_mmlu_prompt(question_text, choices)
                self.log_mismatch(
                    prompt_text,
                    ans,
                    prediction,
                    extracted or pred_letter,
                )

            # Return in Combined schema (benchmark_type added below)
            base_result = (question_text, prediction, ans, uni_score, cost)
            # Still support additional fields downstream if needed, but CSV uses base_result
            result = base_result
            # Store results for aggregation
            self.results[benchmark_type]["scores"].append(uni_score)
            self.results[benchmark_type]["costs"].append(cost)
            return (benchmark_type,) + result
        
        try:
            result = await benchmark.evaluate_problem(formatted_problem, graph)
            score = result[3]  # score is at index 3
            cost = result[4]   # cost is at index 4
            
            # Store results for aggregation
            self.results[benchmark_type]["scores"].append(score)
            self.results[benchmark_type]["costs"].append(cost)
            
            # Add benchmark_type to the result tuple
            # Ensure fixed-width output for CSV schema
            if len(result) > 5:
                result = result[:5]
            return (benchmark_type,) + result
            
        except Exception as e:
            logger.error(f"Error evaluating {benchmark_type} problem: {e}")
            return benchmark_type, problem.get("question", ""), str(e), "", 0, 0.0

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        # This method is not used in the combined benchmark
        # Individual benchmarks handle their own scoring
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["benchmark_type", "question", "prediction", "expected_output", "score", "cost"]

    def get_aggregated_results(self) -> Dict[str, Dict[str, float]]:
        """Calculate aggregated results for each benchmark type"""
        aggregated = {}
        
        for benchmark_type, results in self.results.items():
            scores = results["scores"]
            costs = results["costs"]
            
            if scores:
                avg_score = sum(scores) / len(scores)
                total_cost = sum(costs)
                avg_cost = total_cost / len(costs) if costs else 0
                
                aggregated[benchmark_type] = {
                    "avg_score": avg_score,
                    "total_cost": total_cost,
                    "avg_cost": avg_cost,
                    "sample_count": len(scores)
                }
        
        return aggregated

    async def run_evaluation(self, agent: Callable, va_list: List[int], max_concurrent_tasks: int = 50):
        data = await self.load_data(va_list)
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        
        # Save detailed results
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        
        # Get and log aggregated results
        aggregated = self.get_aggregated_results()
        
        logger.info(f"Combined benchmark results:")
        for benchmark_type, metrics in aggregated.items():
            logger.info(f"  {benchmark_type}: Score={metrics['avg_score']:.5f}, "
                       f"Cost={metrics['total_cost']:.5f}, Samples={metrics['sample_count']}")
        
        # Calculate overall metrics
        overall_score = sum(metrics['avg_score'] * metrics['sample_count'] 
                          for metrics in aggregated.values())
        total_samples = sum(metrics['sample_count'] for metrics in aggregated.values())
        overall_avg_score = overall_score / total_samples if total_samples > 0 else 0
        
        logger.info(f"Overall average score: {overall_avg_score:.5f}")
        
        return overall_avg_score, average_cost, total_cost
