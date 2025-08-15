# -*- coding: utf-8 -*-
# Adapted for AFlow's model and code repository structure
# Core parsing methods kept the same for fair baseline comparison

import torch
from typing import Tuple, Any, List, Callable
import re
import os
import asyncio
import threading
import time
import json
import aiofiles
from math import isclose
import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from scripts.utils.sanitize import sanitize
from scripts.logs import logger
from tqdm.asyncio import tqdm_asyncio
from benchmarks.benchmark import BaseBenchmark


class CombinedBenchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_data_path(self, dataset: str, test: bool) -> str:
        # We override this to prevent it from looking for a file
        # since we will be passing the data directly
        return ""

    async def load_data(
        self, data: List[dict], specific_indices: List[int] = None
    ) -> List[dict]:
        # We override this to accept the in-memory dataset
        if specific_indices is not None:
            return [data[i] for i in specific_indices if i < len(data)]
        return data

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        question_type = problem.get("type", "unknown")
        
        if question_type == "mmlu":
            return await self._evaluate_mmlu_problem(problem, graph)
        elif question_type == "math":
            return await self._evaluate_math_problem(problem, graph)
        elif question_type == "humaneval":
            return await self._evaluate_humaneval_problem(problem, graph)
        else:
            logger.error(f"Unknown question type: {question_type}")
            return ("", "", "", 0.0, 0.0)

    async def _evaluate_mmlu_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        question = problem["question"]
        choices = problem["choices"]
        answer = problem["answer"]

        # Format MMLU prompt with 0-based numbered choices
        formatted_prompt = self._format_mmlu_prompt(question, choices)
        
        # Add instruction for numeric response
        instruction = f"Reply with only the index number (0..{max(0, len(choices)-1)}). No explanation."
        full_prompt = formatted_prompt + "\n\n" + instruction

        try:
            prediction, cost = await graph(full_prompt)
            
            # Extract answer using the same parsing logic as other models
            extracted_answer = self._map_prediction_to_choice_letter(prediction, choices)
            score = 1.0 if (isinstance(answer, int) and 0 <= answer < len(choices) and extracted_answer == choices[answer]) else 0.0

            # Log final MMLU answer and cost
            try:
                logger.info(
                    f"[ANS][MMLU] extracted={extracted_answer} score={score:.1f} "
                    f"cost=${cost:.6f}"
                )
            except Exception:
                pass

            return (
                question,
                prediction,
                answer,
                score,
                cost,
            )

        except Exception as e:
            logger.error(f"Error evaluating MMLU problem: {e}")
            return (question, str(e), answer, 0.0, 0.0)

    async def _evaluate_math_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            result = await graph(input_text)
            
            # Handle both tuple and dict returns
            if isinstance(result, tuple) and len(result) >= 2:
                output, cost = result[0], result[1]
            elif isinstance(result, dict):
                output = result.get('response', '')
                cost = result.get('cost', 0.0)
            else:
                output = str(result)
                cost = 0.0
                
            # Normalize potential non-string outputs early
            output = self._extract_string_from_output(output)

            # Ensure final answer is presented in LaTeX $$ ... $$ with \boxed{...}
            try:
                predicted_for_format = self.extract_model_answer(output)
                if isinstance(predicted_for_format, str):
                    predicted_for_format = predicted_for_format.strip()
                if predicted_for_format:
                    # Only append a canonical $$\boxed{...}$$ line if not already present
                    if "\\boxed{" not in output:
                        output = f"{output.rstrip()}\n\n$$\\boxed{{{predicted_for_format}}}$$"
                    elif "$$" not in output:
                        # Boxed present but not wrapped in $$ ... $$; wrap a final canonical line
                        output = f"{output.rstrip()}\n\n$$\\boxed{{{predicted_for_format}}}$$"
            except Exception:
                # If formatting fails for any reason, proceed with original output
                pass

            if not output:
                raise ValueError("output is empty")

            uni_score, extracted_output = self._calculate_math_score(expected_output, output)

            if uni_score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output,
                    extract_answer_code=self.get_function_code(
                        self.extract_model_answer
                    ),
                )

            # Log final MATH answer and cost
            try:
                logger.info(
                    f"[ANS][MATH] extracted={extracted_output} score={uni_score} "
                    f"cost=${cost:.6f}"
                )
            except Exception:
                pass

            return (
                input_text,
                output,
                expected_output,
                uni_score,
                cost,
            )

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return (
                input_text,
                str(e),
                expected_output,
                0.0,
                0.0,
            )

    async def _evaluate_humaneval_problem(self, problem: dict, graph: Callable) -> Tuple[Any, ...]:
        input_text = problem["prompt"]
        expected_output = (
            "\nCorrect Solution:\ndef "
            + problem["entry_point"]
            + "(params you should put here):"
            + "\n\n"
            + problem["canonical_solution"]
        )

        try:
            result = await graph(input_text, problem["entry_point"])
            
            # Handle both tuple and dict returns
            if isinstance(result, tuple) and len(result) >= 2:
                prediction, cost = result[0], result[1]
            elif isinstance(result, dict):
                prediction = result.get('response', '')
                cost = result.get('cost', 0.0)
            else:
                prediction = str(result)
                cost = 0.0

            if not prediction:
                raise ValueError("Prediction is empty")

            # Normalize complex outputs to a string (raw code or text)
            prediction = self._extract_string_from_output(prediction)

            ret = self._check_humaneval_solution(prediction, problem["test"], problem["entry_point"])
            if not isinstance(ret, (list, tuple)) or len(ret) < 2:
                logger.info("Invalid return value from check_solution.")
            test_case_details = ret[1]
            expected_output = test_case_details + expected_output
            score = 1.0 if ret[0] == self.PASS else 0.0
            if score == 0:
                self.log_mismatch(input_text, expected_output, prediction, score)

            # Log HumanEval result and cost
            try:
                outcome = "PASS" if score == 1.0 else "FAIL"
                logger.info(
                    f"[ANS][HumanEval] entry_point={problem.get('entry_point','')} result={outcome} "
                    f"cost=${cost:.6f}"
                )
            except Exception:
                pass

            return (
                input_text,
                prediction,
                expected_output,
                score,
                cost,
            )

        except asyncio.TimeoutError:
            logger.info("Timeout error. Skipping this sample.")
            return (
                input_text,
                "Timeout",
                expected_output,
                0.0,
                0.0,
            )

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return (
                input_text,
                "Timeout",
                expected_output,
                0.0,
                0.0,
            )

    def _format_mmlu_prompt(self, question: str, choices: Any) -> str:
        """Format MMLU prompt with 0-based numbered choices - same as other models"""
        if isinstance(choices, list) and choices:
            lines = []
            for i, choice in enumerate(choices):
                # Present as 0. Choice, 1. Choice, etc.
                lines.append(f"{i}. {choice}")
            return question.strip() + "\n\n" + "\n".join(lines)
        return question.strip()

    def _map_prediction_to_choice_letter(self, prediction: str, choices: Any) -> str:
        """Map prediction to choice letter - same parsing logic as other models"""
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Try to parse as a 0-based integer index first
        try:
            num_match = re.search(r"\b(\d+)\b", prediction)
            if num_match:
                idx = int(num_match.group(1))
                if 0 <= idx < len(choices):
                    return choices[idx]
        except (ValueError, IndexError):
            pass

        # Try to parse as a 1-based integer index
        try:
            num_match = re.search(r"\b(\d+)\b", prediction)
            if num_match:
                idx = int(num_match.group(1)) - 1  # Convert to 0-based
                if 0 <= idx < len(choices):
                    return choices[idx]
        except (ValueError, IndexError):
            pass

        # Try exact match for A-Z
        for i, label in enumerate(labels[:len(choices)]):
            if prediction.strip().upper() == label:
                return choices[i]

        # Try fuzzy match for A-Z
        for i, label in enumerate(labels[:len(choices)]):
            if label in prediction.upper():
                return choices[i]

        # Fallback to first choice
        return choices[0] if choices else ""

    def _extract_string_from_output(self, output):
        """Extract a reasonable string from various output formats - same as other models"""
        # direct string
        if isinstance(output, str):
            return output
        # list/tuple: prefer last non-empty string, else last item stringified
        if isinstance(output, (list, tuple)):
            for item in reversed(output):
                if isinstance(item, str) and item.strip():
                    return item
            return str(output[-1]) if output else ""
        # dict: try common keys
        if isinstance(output, dict):
            for key in ["answer", "solution", "result", "output", "content", "text"]:
                val = output.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            return json.dumps(output, ensure_ascii=False)
        # fallback: stringify
        return str(output)

    def _calculate_math_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        """Calculate math score - same logic as other models"""
        # Ensure inputs are strings
        if not isinstance(expected_output, str):
            expected_output = str(expected_output)
        if not isinstance(prediction, str):
            prediction = str(prediction)
            
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def extract_model_answer(self, text: str) -> str:
        """Extract model answer - same logic as other models"""
        # Ensure text is a string before processing
        if not isinstance(text, str):
            text = self._extract_string_from_output(text)

        # Prefer explicit XML-like tags commonly emitted by operators
        tag_patterns = [
            r"<final_answer>\s*([\s\S]*?)\s*</final_answer>",
            r"<answer>\s*([\s\S]*?)\s*</answer>",
            r"<refined>\s*([\s\S]*?)\s*</refined>",
            r"<refined_answer>\s*([\s\S]*?)\s*</refined_answer>",
        ]
        for p in tag_patterns:
            m = re.findall(p, text, re.IGNORECASE)
            if m:
                candidate = m[-1].strip()
                # If tag contains extra markup, strip code fences/backticks
                candidate = re.sub(r"^```[a-zA-Z]*\\n|```$", "", candidate).strip().strip("`")
                if candidate:
                    return candidate

        # Then prefer last boxed expression
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # Fallback: last sentence
        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        """Check if math expressions are equal - same logic as other models"""
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        """Check if string represents a digit - same logic as other models"""
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        """Parse digits from string - same logic as other models"""
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        """Check symbolic equality - same logic as other models"""
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        """Get function source code - same logic as other models"""
        try:
            import inspect
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_math_output(self, graph, input_text):
        """Generate math output with retry - same logic as other models"""
        return await asyncio.wait_for(graph(input_text), timeout=1500)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, args, timeout):
        """Run function with timeout - same logic as other models"""
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def _check_humaneval_solution(self, solution, test, entry_point):
        """Check HumanEval solution - same logic as other models"""
        try:
            from typing import List, Dict, Tuple, Optional, Any
            
            # Sanitize and normalize model output to executable Python code
            solution = sanitize(code=solution, entrypoint=entry_point)

            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            # Known helper injections for specific problems (compatibility)
            if entry_point == "decode_cyclic":
                solution = (
                    '\n\ndef encode_cyclic(s: str):\n    """\n    returns encoded string by cycling groups of three characters.\n    """\n    # split string to groups. Each of length 3.\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n    # cycle elements in each group. Unless group has fewer elements than 3.\n    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)'
                    + "\n\n"
                    + solution
                )
            elif entry_point == "decode_shift":
                solution = (
                    '\n\ndef encode_shift(s: str):\n    """\n    returns encoded string by shifting every character by 5 in the alphabet.\n    """\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n\n'
                    + solution
                )
            elif entry_point == "find_zero":
                solution = (
                    "\n\ndef poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n"
                    + solution
                )

            # Execute the candidate solution
            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(
                    f"Function {entry_point} is not defined in the solution."
                )

            # Load and run unit tests (they must define a `check` callable)
            exec(test, global_dict)
            check = global_dict["check"]

            # Run tests with timeout against the generated function
            result = self.run_with_timeout(check, (global_dict[entry_point],), 15)
            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

            return result

        except self.TimeoutError:
            return (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as e:
            error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
            return (self.FAIL, error_message)

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_fixed(1),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _generate_humaneval_output(self, graph, prompt, entry_point):
        """Generate HumanEval output with retry - same logic as other models"""
        # Generate output with a timeout of 200 seconds
        return await asyncio.wait_for(graph(prompt, entry_point), timeout=1500)

    def calculate_score(self, expected_output: Any, prediction: Any) -> Tuple[float, Any]:
        """
        Unified scoring method that can handle all problem types - same logic as other models
        """
        # For MMLU problems, expected_output is the answer index, prediction is the model output
        if isinstance(expected_output, int) and isinstance(prediction, str):
            # This would need to be adapted based on how MMLU is handled in the specific implementation
            return 0.0, prediction
        
        # For MATH problems, both are strings
        elif isinstance(expected_output, str) and isinstance(prediction, str):
            score, extracted = self._calculate_math_score(expected_output, prediction)
            return float(score), extracted
        
        # For HumanEval problems, expected_output is the test result, prediction is the code
        elif isinstance(expected_output, (bool, tuple)) and isinstance(prediction, str):
            # HumanEval scoring is handled in _evaluate_humaneval_problem
            # This is a fallback for the abstract method
            return 0.0, prediction
        
        # Fallback for unknown types
        else:
            logger.warning(f"Unknown scoring types: expected_output={type(expected_output)}, prediction={type(prediction)}")
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        """Get result columns - adapted for AFlow structure"""
        return [
            "question",
            "prediction", 
            "expected_answer",
            "score",
            "cost",
        ]

    async def run_evaluation(
        self,
        graph: Callable,
        va_list: List[int],
        is_test: bool,
        sample: int,
        is_textgrad: bool = False,
        max_concurrent_tasks: int = 1,
    ):
        """Run evaluation - adapted for AFlow structure"""
        import time
        start_time = time.time()
        
        # Load data from file
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        
        # Filter by va_list
        loaded_data = await self.load_data(data, va_list)

        if is_test:
            # Use rate-limited evaluation for test
            tasks = [self.evaluate_problem(problem, graph) for problem in loaded_data]
            results = await tqdm_asyncio.gather(
                *tasks, desc=f"Evaluating {self.name} problems", total=len(loaded_data)
            )
        else:
            # Use base class evaluation
            results = await self.evaluate_all_problems(
                loaded_data,
                graph,
                max_concurrent_tasks=1,
                repetitions=1,
                is_textgrad=is_textgrad,
            )

        columns = self.get_result_columns()
        average_score = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")

        # Aggregate cost info by summing per-problem deltas from results
        total_cost = sum(float(res[4]) for res in results if len(res) > 4)
        num_results = len(results)
        average_cost = total_cost / num_results if num_results > 0 else 0.0

        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / num_results if num_results > 0 else 0.0

        # Print aggregate cost stats
        try:
            logger.info(
                f"Total Cost: ${total_cost:.6f} | Avg Cost/sample: ${average_cost:.6f}"
            )
        except Exception:
            logger.info(
                f"Total Cost: ${total_cost:.6f} | Avg Cost/sample: ${average_cost:.6f}"
            )

        # Return tuple format expected by evaluation_utils: (average_score, average_cost, total_tokens)
        # Note: total_tokens is set to 0 since AFlow doesn't track tokens in the same way
        return average_score, average_cost, 0 