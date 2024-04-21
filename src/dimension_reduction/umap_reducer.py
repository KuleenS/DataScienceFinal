from typing import List
import umap

from dimension_reduction import DimensionReduction

class UMAPReducer(DimensionReduction):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.reducer = umap.UMAP(**kwargs)
    
    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        return self.reducer.fit_transform(points)