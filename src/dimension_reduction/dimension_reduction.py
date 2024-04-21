from abc import ABC

from typing import List

class DimensionReduction(ABC):

    def __init__(self, kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs

    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        pass