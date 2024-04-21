from typing import List
from sklearn.decomposition import KernelPCA

from dimension_reduction import DimensionReduction


class KernelPCAReducer(DimensionReduction):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.reducer = KernelPCA(**kwargs)
    
    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        return self.reducer.fit_transform(points)