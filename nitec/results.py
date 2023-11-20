from dataclasses import dataclass
import numpy as np

@dataclass
class NITECResultContainer:

    results: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray