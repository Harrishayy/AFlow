import json
import os
import re
import time
import traceback
from typing import List

from scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE,
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_TEMPLATE,
)
from scripts.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except ImportError as e:
            logger.error(f"Error loading graph for round {round_number}: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.error(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: str,
    ) -> str:
        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.error(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        def _strip_code_fences(text: str) -> str:
            # Remove markdown code fences like ```python ... ``` or plain ```
            text = re.sub(r"```[ \t]*[a-zA-Z0-9_-]*[ \t]*\n?", "", text)
            text = text.replace("```", "")
            return text

        def _make_valid_prompt_module(raw: str, referenced_names=None) -> str:
            """Return a syntactically valid Python module text for prompt.py.
            If raw already looks like valid Python, keep it; otherwise wrap as a constant.
            """
            header = "# -*- coding: utf-8 -*-\n"
            body = (raw or "").strip()
            # Fast path: accept if compiles
            try:
                compile(body or "CUSTOM_PROMPT = ''\n", "<prompt>\n", "exec")
                # Ensure we don't end up with an empty file
                return header + (body if body else "CUSTOM_PROMPT = ''\n")
            except Exception:
                # Wrap entire content into named constants referenced by graph when available
                lines = []
                names = list(referenced_names or [])
                if names:
                    for name in names:
                        lines.append(f"{name} = {repr(raw or '')}\n")
                else:
                    lines.append(f"CUSTOM_PROMPT = {repr(raw or '')}\n")
                wrapped = "".join(lines)
                try:
                    compile(wrapped, "<prompt>", "exec")
                    return header + wrapped
                except Exception:
                    # Last resort
                    return header + "CUSTOM_PROMPT = ''\n"

        graph_body = _strip_code_fences(response.get("graph", ""))
        # Remove unsupported/foreign imports the model might hallucinate
        graph_body = re.sub(r"^\s*from\s+core\.[^\n]*\n", "", graph_body, flags=re.MULTILINE)
        graph_body = re.sub(r"^\s*import\s+core\.[^\n]*\n", "", graph_body, flags=re.MULTILINE)
        graph_body = re.sub(r"^\s*import\s+metagpt[^\n]*\n", "", graph_body, flags=re.MULTILINE)
        graph_body = re.sub(r"^\s*from\s+metagpt[^\n]*\n", "", graph_body, flags=re.MULTILINE)
        # Ensure operators are constructed with llm when missing
        graph_body = re.sub(r"operator\.ScEnsemble\(\s*\)", "operator.ScEnsemble(self.llm)", graph_body)
        graph_body = re.sub(r"operator\.AnswerGenerate\(\s*\)", "operator.AnswerGenerate(self.llm)", graph_body)
        graph_body = re.sub(r"operator\.CustomCodeGenerate\(\s*\)", "operator.CustomCodeGenerate(self.llm)", graph_body)
        graph_body = re.sub(r"operator\.Custom\(\s*\)", "operator.Custom(self.llm)", graph_body)
        graph_body = re.sub(r"operator\.Test\(\s*\)", "operator.Test(self.llm)", graph_body)
        graph = WORKFLOW_TEMPLATE.format(graph=graph_body, round=round_number, dataset=dataset)

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        prompt_path = os.path.join(directory, "prompt.py")
        prompt_raw = _strip_code_fences(response.get("prompt", ""))
        # Determine referenced prompt constants from graph before building prompt module
        referenced = set(re.findall(r"prompt_custom\.([A-Z_][A-Z0-9_]*)", graph_body))
        prompt_module = _make_valid_prompt_module(prompt_raw, referenced)
        with open(prompt_path, "w", encoding="utf-8") as file:
            file.write(prompt_module)

        # Ensure any prompt_custom.CONSTANTS referenced by graph exist in prompt.py
        try:
            missing = []
            # Re-use referenced collected above
            existing = set(re.findall(r"^([A-Z_][A-Z0-9_]*)\s*=", prompt_module, flags=re.MULTILINE))
            for name in referenced:
                if name not in existing:
                    missing.append(name)
            # Always ensure commonly used constants exist
            always_defaults = ["ANALYSIS_PROMPT"]
            for name in always_defaults:
                if name not in existing and name not in missing:
                    missing.append(name)
            if missing:
                with open(prompt_path, "a", encoding="utf-8") as f:
                    f.write("\n\n# Auto-added missing prompt constants to satisfy graph references\n")
                    for name in missing:
                        f.write(f"{name} = \"\"\n")
        except Exception:
            # Best-effort; don't fail graph writing
            pass

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
