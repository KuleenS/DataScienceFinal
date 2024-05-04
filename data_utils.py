from datasets import load_dataset
import sys
import json
from tqdm import tqdm

def load_cultural_bank(save_path):
    ds = load_dataset("SALT-NLP/CultureBank")
    reddit_data = ds['reddit']
    tiktok_data = ds['tiktok']

    # concatenate the data for SentenceBERT
    columns = ['context', 'goal', 'relation', 'actor', 'actor_behavior', 'recipient', "recipient_behavior", 'other_descriptions']
    reddit_concat, tiktok_concat = [""] * len(reddit_data), [""] * len(tiktok_data)
    
    for col in columns:
        data = reddit_data[col]
        data = [col + ": " + str(d) if d is not None else "" for d in data ]
        assert len(data) == len(reddit_concat)
        if col == columns[0]:
            reddit_concat = data
        else:
            reddit_concat = [reddit_concat[i] + ", " + data[i] for i in range(len(data))]
        # tiktok
        data = tiktok_data[col]
        data = [col + ": " + str(d) if d is not None else "" for d in data ]
        assert len(data) == len(tiktok_concat)
        if col == columns[0]:
            tiktok_concat = data
        else:
            tiktok_concat = [tiktok_concat[i] + ", " + data[i] for i in range(len(data))]
    
    cultural_groups = reddit_data['cultural group']
    topics = reddit_data['topic']
    # save {'id': i, 'contents': reddit_concat[i]} in jsonl format
    with open(save_path + "/reddit.jsonl", "w") as f:
        for i in range(len(reddit_concat)):
            obj = {'id': i, 'contents': reddit_concat[i], 'cultural_group': cultural_groups[i], 'topic': topics[i]}
            f.write(json.dumps(obj) + "\n")
        f.close()

    cultural_groups = tiktok_data['cultural group']
    topics = tiktok_data['topic']
    with open(save_path + "/tiktok.jsonl", "w") as f:
        for i in range(len(tiktok_concat)):
            obj = {'id': i, 'contents': tiktok_concat[i], 'cultural_group': cultural_groups[i], 'topic': topics[i]}
            f.write(json.dumps(obj) + "\n")
        f.close()

def read_cultural_bank(path):
    topics = []
    cultural_groups = []
    data = []
    # read jsonl
    with open(path) as f:
        for line in tqdm(f.readlines(), desc="Reading data", total=len(f.readlines())):
            obj = json.loads(line)
            data.append(obj['contents'])
            topics.append(obj['topic'])
            cultural_groups.append(obj['cultural_group'])                
        f.close()
    return data, topics, cultural_groups

if __name__ == "__main__":
    path = sys.argv[1]
    func = sys.argv[2]
    if func == "load":
        load_cultural_bank(path)
    elif func == "read":
        read_cultural_bank(path)

    
