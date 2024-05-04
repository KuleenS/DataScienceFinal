from sentence_transformers import SentenceTransformer
import argparse
import torch
from tqdm import tqdm
import json
import pickle
from data_utils import read_cultural_bank

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="The path to the data")
    parser.add_argument("--model_name", type=str, help="The model name for SentenceTransformer")
    parser.add_argument("--bsz", type=int, default=128)
    args = parser.parse_args()
    data_path = args.data_path
    model_name_for_path = args.model_name.split("/")[-1]
    save_path = data_path.split(".")[0] + f"-model_name={model_name_for_path}-bsz={args.bsz}.pkl"

    sbert = SentenceTransformer(args.model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    data, _, _ = read_cultural_bank(data_path)
    
    # batch encode data
    embeddings = []
    batches = [data[i:i+args.bsz] for i in range(0, len(data), args.bsz)]
    assert len(data) == sum([len(b) for b in batches])
    for batch in tqdm(batches, desc="Encoding data"):
        embeddings.extend(sbert.encode([d[0] for d in batch]))
    # save embeddings
    with open(save_path, "wb") as f:
        pickle.dump(embeddings, f)
        f.close()

