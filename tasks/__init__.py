from .paradetox import ParaDetoxProbInferenceForStyle
from .jailbreak import JailBreakProbInferenceForStyle
from .demo import DemoProbInferenceForStyle
from .base import RewardProbInference

task_mapper = {
    "paradetox": ParaDetoxProbInferenceForStyle,
    "jailbreak": JailBreakProbInferenceForStyle,
    'demo': DemoProbInferenceForStyle,
    "reward": RewardProbInference
}


def load_task(name):
    if name not in task_mapper.keys():
        raise ValueError(f"Unrecognized dataset `{name}`")

    return task_mapper[name]
