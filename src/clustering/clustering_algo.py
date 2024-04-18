
from abc import ABC

from typing import List, Dict, Any

class ClusteringAlgorithm(ABC):

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        super().__init__()

        self.kwargs = kwargs

    def cluster(self, points: List[List[float]]) -> List[int]:
        pass