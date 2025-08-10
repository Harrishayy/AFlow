import re
from typing import Any, Callable, List, Tuple

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class MMLUBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def _extract_predicted_index(self, text: str) -> int | None:
        """Extract a predicted option index from model output.
        Supports numeric indices (0-based preferred, falls back to 1-based),
        and letter choices (A->0, B->1, ...).
        """
        # Try numeric first
        mnum = re.search(r"\b(\d{1,2})\b", text)
        if mnum:
            try:
                raw = int(mnum.group(1))
                # Prefer 0-based; if looks like 1-based, caller can still compare
                return raw if raw >= 0 else None
            except Exception:
                pass
        # Then letters
        patterns = [
            r"The final answer is:\s*([A-Z])",
            r"Answer:\s*([A-Z])",
            r"The answer is\s*([A-Z])",
            r"([A-Z])\s*is the correct answer",
            r"Option\s*([A-Z])",
            r"Choice\s*([A-Z])",
            r"\b([A-Z])\s*\)",
            r"\b([A-Z])\.",
            r"\b([A-Z])\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()
                if "A" <= letter <= "Z":
                    return ord(letter) - ord("A")
        return None

    def calculate_score(self, expected_output: Any, prediction: str) -> Tuple[int, str]:
        """Compare predicted option against expected output.
        Supports:
        - Model output as number (0-based preferred; 1-based also accepted)
        - Model output as letter (A/B/C/..)
        - Expected output as 0-based integer or numeric string or letter
        """
        predicted_index = self._extract_predicted_index(prediction)
        if predicted_index is None:
            return 0, ""

        # Derive expected index
        expected_index = None
        if isinstance(expected_output, int):
            expected_index = expected_output
        elif isinstance(expected_output, str):
            eo = expected_output.strip()
            if eo.isdigit():
                try:
                    expected_index = int(eo)
                except Exception:
                    expected_index = None
            elif len(eo) == 1 and "A" <= eo.upper() <= "Z":
                expected_index = ord(eo.upper()) - ord("A")
            else:
                m = re.search(r"([A-Z])", eo, re.IGNORECASE)
                if m:
                    expected_index = ord(m.group(1).upper()) - ord("A")

        if expected_index is None:
            return 0, str(predicted_index)

        # Also accept 1-based model outputs implicitly
        score = 1 if (predicted_index == expected_index or predicted_index - 1 == expected_index) else 0
        return score, str(predicted_index)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        question = problem.get("question", "")
        choices = problem.get("choices", [])
        expected_output = problem.get("answer", "")

        # Present options as 0..N-1 and instruct numeric-only reply
        if isinstance(choices, list) and choices:
            lines = [f"{i}. {c}" for i, c in enumerate(choices)]
            input_text = question.strip() + "\n\n" + "\n".join(lines) + "\n\nReply with only the index number (0.." + str(max(0, len(choices)-1)) + ")."
        else:
            input_text = question

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)
            
            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(self.extract_model_answer),
                )
            
            return input_text, output, expected_output, uni_score, cost
            
        except Exception as e:
            logger.info(f"Error evaluating MMLU problem: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_function_code(self, func):
        try:
            import inspect
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
