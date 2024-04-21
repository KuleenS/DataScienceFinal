from typing import List
from sklearn.manifold import LocallyLinearEmbedding

from dimension_reduction import DimensionReduction


class LocalLinearEmbeddingReducer(DimensionReduction):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.reducer = LocallyLinearEmbedding(**kwargs)
    
    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        return self.reducer.fit_transform(points)