from typing import Dict, Any, List, Tuple

import numpy as np

from src.datasets.dataset import Dataset

class MoonsSyntheticDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
    
    def generate_dataset(self, kwargs: Dict[str, Any]) -> Tuple[List[Any], List[float]]:
       rng = np.random.RandomState()
       
       return rng.rand(**kwargs, 2), None



