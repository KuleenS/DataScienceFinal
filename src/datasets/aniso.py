from typing import Dict, Any, List, Tuple

import numpy as np

import sklearn.datasets as datasets

from src.datasets.dataset import Dataset

class AnisoSyntheticDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
    
    def generate_dataset(self, kwargs: Dict[str, Any]) -> Tuple[List[Any], List[float]]:
        X, y = datasets.make_blobs(**kwargs)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]

        X_aniso = np.dot(X, transformation)

        return X_aniso, y








