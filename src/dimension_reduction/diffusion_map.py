from typing import List
from pydiffmap.diffusion_map import DiffusionMap

from dimension_reduction import DimensionReduction

class DiffusionMapReducer(DimensionReduction):

    def __init__(self, kwargs) -> None:
        super().__init__(kwargs)

        self.reducer = DiffusionMap.from_sklearn(**kwargs)
    
    def reduce(self, points: List[List[float]]) -> List[List[float]]:
        return self.reducer.fit_transform(points)

