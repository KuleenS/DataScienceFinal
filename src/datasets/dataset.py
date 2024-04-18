from abc import ABC

from typing import Dict, Any, List, Tuple

class Dataset(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def generate_dataset(self) -> Tuple[List[Any], List[float]]:
        pass