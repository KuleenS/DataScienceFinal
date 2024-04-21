from typing import List
from sklearn.manifold import SpectralEmbedding

from dimension_reduction import DimensionReduction

class MultiDimensionalScalingReducer(DimensionReduction):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.reducer = SpectralEmbedding(**kwargs)
    
    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        return self.reducer.fit_transform(points)