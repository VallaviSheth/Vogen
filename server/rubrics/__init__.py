from .critic_score import CriticScoreRubric
from .justification import JustificationRubric
from .novelty import NoveltyRubric
from .difficulty import DifficultyRubric
from .calibration import CalibrationRubric

RUBRICS = {
    "critic_score": CriticScoreRubric,
    "justification": JustificationRubric,
    "novelty": NoveltyRubric,
    "difficulty": DifficultyRubric,
    "calibration": CalibrationRubric,
}