from typing import List
from sklearn.decomposition import PCA

from dimension_reduction import DimensionReduction


class PCAReducer(DimensionReduction):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.reducer = PCA(**kwargs)
    
    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        return self.reducer.fit_transform(points)