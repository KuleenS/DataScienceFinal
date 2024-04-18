from abc import ABC

class Dataset(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    def generate_dataset(self):
        pass