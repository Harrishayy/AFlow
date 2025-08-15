from typing import Literal
import sys
import os
import importlib.util

# Load template operator from local path (avoid stdlib operator)
template_dir = os.path.join(os.path.dirname(__file__), '..', 'template')
operator_file = os.path.join(template_dir, 'operator.py')
if template_dir not in sys.path:
    sys.path.insert(0, template_dir)
spec = importlib.util.spec_from_file_location("combined_template_operator", operator_file)
custom_operator = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(custom_operator)
# Back-compat alias
operator = custom_operator

from scripts.async_llm import AsyncLLM
import workspace.Combined.workflows.round_1.prompt as prompt_custom
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
        # Pre-instantiate operators from operator.json
        self.cot = operator.CoT(self.llm)
        self.debate = operator.Debate(self.llm)
        self.self_consistency = operator.SelfConsistency(self.llm)
        self.self_refine = operator.SelfRefine(self.llm)
        self.ensemble = operator.Ensemble(self.llm)
        self.testing = operator.Testing(self.llm)
        # self.react = operator.ReAct(self.llm)  # Disabled for this benchmark
        self.early_exit = operator.EarlyExit(self.llm)

    async def __call__(self, problem: str, entry_point: str = ""):
        """
        Simple Chain-of-Thought approach for all problem types.
        Returns (solution, cost).
        """
        # Use CoT operator for all problem types (baseline)
        resp = await self.cot(problem=problem)
        return resp.get('response', ''), self.llm.get_usage_summary()["total_cost"]
    
