from experiments.datasets.Combined.train.template.operator import (
    CoT, Debate, SelfConsistency, SelfRefine, Ensemble,
    Testing, # ReAct, 
    EarlyExit
)

operator_mapping = {
    "CoT": CoT,
    "Debate": Debate,
    "SelfConsistency": SelfConsistency,
    "SelfRefine": SelfRefine,
    "Ensemble": Ensemble,
    "Testing": Testing,
    # "ReAct": ReAct,
    "EarlyExit": EarlyExit,
}

operator_names = list(operator_mapping.keys())