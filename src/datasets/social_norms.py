from typing import Any, List, Tuple

import pandas as pd

from sentence_transformers import SentenceTransformer

import torch

from dataset import Dataset

class SocialNormsDataset(Dataset):

    def __init__(self, mode: str, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()

        self.modes = set(["hypothesis", "premise", "together"])

        if mode not in self.modes:
            raise ValueError(f"{mode} not in {self.mode}")

        self.mode = mode

        self.data = pd.read_csv("data/data.tsv", sep="\t")

        self.model_name = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def batch(self, iterable, n=32):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def generate_dataset(self) -> Tuple[List[Any], List[float]]:
        
        embeddings = []

        labels = list(zip(list(self.data["us_ratings"]), list(self.data["in_ratings"])))

        sentences = None

        if self.mode == "hypothesis":
            sentences = list(self.data["hypothesis"])

        elif self.mode == "premise":
            sentences = list(self.data["premise"])

        else:
            sentences = list(self.data["premise"] + self.data["hypothesis"])
            
        batched_sentences = self.batch(sentences)

        for batch in batched_sentences:
            batch_embeddings = self.model.encode(batch, convert_to_numpy=False, convert_to_tensor=True, normalize_embeddings=True)

            embeddings.extend(batch_embeddings)

        return list(zip(embeddings, labels))