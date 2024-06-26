from typing import Dict, Any, List, Tuple

import sklearn.datasets as datasets

from src.datasets.dataset import Dataset

class CirclesSyntheticDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
    
    def generate_dataset(self, kwargs: Dict[str, Any]) -> Tuple[List[Any], List[float]]:
        X, y = datasets.make_circles(**kwargs)

        return X, y








