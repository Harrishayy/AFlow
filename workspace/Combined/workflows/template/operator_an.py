from pydantic import BaseModel, Field
from typing import List

class CoTOp(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning")
    final_answer: str = Field(description="Final answer")

class DebateTurn(BaseModel):
    debater_id: int = Field(description="0-based debater index")
    statement: str = Field(description="Statement for this turn")

class DebateOp(BaseModel):
    rounds: List[List[DebateTurn]] = Field(description="Debate rounds")
    final_answer: str = Field(description="Final answer after debate")

class SelfConsistencyOp(BaseModel):
    answers: List[str] = Field(description="Five CoT answers")
    voted_answer: str = Field(description="Majority-voted answer")

class SelfRefineOp(BaseModel):
    iterations: List[str] = Field(description="Refinement notes up to five iterations")
    refined_answer: str = Field(description="Refined final answer")

class EnsembleMember(BaseModel):
    source: str = Field(description="LLM/agent source")
    answer: str = Field(description="Answer from this member")

class EnsembleOp(BaseModel):
    members: List[EnsembleMember] = Field(description="Three members")
    final_answer: str = Field(description="Aggregated final answer")

class TestingOp(BaseModel):
    tests: List[str] = Field(description="Generated test cases")

class ReActStep(BaseModel):
    thought: str = Field(description="Brief reasoning")
    action: str = Field(description="tool|search|read|execute|math|web|finish")
    input: str = Field(description="input to the chosen tool")
    observation: str = Field(description="result of the step")

class ReActOp(BaseModel):
    steps: List[ReActStep] = Field(description="ReAct trajectory")
    final_answer: str = Field(description="Final answer")

class EarlyExitOp(BaseModel):
    reason: str = Field(description="Why exiting early")
