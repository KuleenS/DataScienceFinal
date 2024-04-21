from typing import List
from sklearn.manifold import TSNE

from dimension_reduction import DimensionReduction


class TSNEReducer(DimensionReduction):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.reducer = TSNE(**kwargs)
    
    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        return self.reducer.fit_transform(points)