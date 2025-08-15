from typing import Literal
import sys
import os
import importlib.util

# Ensure template operator is loaded from local path and not stdlib 'operator'
template_dir = os.path.join(os.path.dirname(__file__), '..', 'template')
operator_file = os.path.join(template_dir, 'operator.py')
if template_dir not in sys.path:
    sys.path.insert(0, template_dir)
spec = importlib.util.spec_from_file_location("combined_template_operator", operator_file)
custom_operator = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(custom_operator)
# Backwards-compat alias so graphs that refer to `operator` still work
operator = custom_operator

import workspace.Combined.workflows.round_6.prompt as prompt_custom
from scripts.async_llm import AsyncLLM, create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = AsyncLLM(llm_config)
        # Pre-instantiate operators
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.test = operator.Test(self.llm)

    # No special MMLU routing; non-code problems go through AnswerGenerate

    async def __call__(self, problem: str, entry_point: str = ""):
        """
        Route by task type:
        - HumanEval (code): if entry_point provided → CustomCodeGenerate
        - Non-code (MATH/MMLU/QA): AnswerGenerate followed by ScEnsemble
        Returns (solution, cost).
        """
        # HumanEval path
        if entry_point:
            code = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            return code.get('response', ''), self.llm.get_usage_summary()["total_cost"]

        # Non-code path (MATH/MMLU/QA): use Custom with per-round prompt_custom
        resp = await self.custom(input=problem, instruction=prompt_custom.NON_CODE_PROMPT)
        solutions = [resp.get('response', '')]
        # Use ScEnsemble to select the most frequent solution
        ensemble_resp = await self.sc_ensemble(solutions=solutions, problem=problem)
        return ensemble_resp.get('response', ''), self.llm.get_usage_summary()["total_cost"]
