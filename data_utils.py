from datasets import load_dataset
import sys
import json
from tqdm import tqdm
from collections import Counter
merged_group_labels = {
    'Americans': ['Americans', 'American', 'US', 'US citizens', 'United States', 'People in the USA', 'North Americans', 'US residents', 'US people', 'US and Canadian people', 'US and Canadian'],
    'Dutch': ['Dutch', 'Dutch people', 'Nederlanders', 'Netherlanders'],
    'French': ['French', 'French people'],
    'Filipinos': ['Filipinos'],
    'British': ['British', 'British people', 'Brits', 'UK', 'UK residents', 'people in the UK', 'UK citizens', 'Britons', 'English', 'English people'],
    'Japanese': ['Japanese', 'Japanese people'],
    'Chinese': ['Chinese', 'Chinese people'],
    'Swedish': ['Swedish', 'Swedes'],
    'Danish': ['Danish', 'Danish people'],
    'Asians': ['Asians', 'Asian', 'Asian men and women', 'Asian men', 'Asian women', 'Asian cultures', 'Asian countries', 'Asian men', 'Asian immigrants in the US'],
    'Turkish': ['Turkish', 'Turkish people', 'Turks'],
    'Jewish': ['Jews', 'Jewish people'],
    'Welsh': ['Welsh', 'Welsh people'],
    'Finns': ['Finns', 'Finnish'],
    'Spanish': ['Spanish', 'Spanish people', 'Spaniards'],
    'Latinos': ['Latinos', 'Hispanic', 'Hispanic people', 'Latinos in the US', 'Hispanic Americans', 'Latinos and Latino Americans', 'Latin America', 'Latin Americans', 'People from Latin America'],
    'Black Americans': ['Black Americans', 'African Americans', 'Black people', 'Black', 'Black and African American people'],
    'Mormons': ['Mormons', 'Mormons and ex-Mormons'],
    'Russians': ['Russians', 'Russian'],
    'Irish': ['Irish', 'Irish people'],
    'Mexicans': ['Mexicans', 'Mexican'],
    'Vietnamese': ['Vietnamese', 'Vietnamese people'],
    'Scandinavians': ['Scandinavians', 'Nordic countries'],
    'Portuguese': ['Portuguese', 'Portuguese people'],
    'Polish': ['Poles', 'Polish', 'Polish people'],
    'Canadians': ['Canadians', 'Canadian', 'Canada', 'Canadian people'],
    'Middle Easterners': ['Middle Easterners', 'Middle Eastern people', 'Arabs'],
    'European': ['European', 'European countries', 'Europeans'],
    'Kpop fans': ['Kpop fans', 'Korean pop music (K-pop) fans'],
    'Portlanders': ['Portlanders', 'Portland residents'],
    'Southerners': ['People in the South', 'Southerners', 'Southern Americans'],
    'Germans': ['Germany', 'German people', 'Germans', 'Germans and immigrants in Germany'],
    'Western': ['Western countries', 'Western cultures'],
    'Czechs': ['Czech Republic', 'Czechs'],
    'Malayalees': ['Malay', 'Malayalees'],
    'Malaysians': ['Malaysians', 'Malaysian', 'Malaysian people', 'Malaysian Chinese'],
    'Québécois': ['Québécois', 'Quebec', 'Québécois people', 'French Canadians'],
    'South Asians': ['South Asians', 'South Asians and South East Asians', 'Southeast Asians'],
    'African': ['African', 'African people', 'Africans'],
    'Rural': ['Rural areas', 'Rural communities', 'people from rural areas', 'rural communities'],
    'Expats': ['people living abroad', 'Expats'],
    'Muslims': ['Muslims', 'Muslim', 'Muslim people', 'British Muslims'],
    'Indians': ['Indians', 'Indian people', 'Indian', 'Indian and Indian-American', 'Indian and Pakistani people', 'Indian and Indian American'],
    'Koreans': ['Koreans', 'Korean people', 'South Koreans', 'South Korean', 'Korean'],
    'Thai': ['Thai', 'Thai people'],
    'Italians': ['Italians', 'Italian people', 'Italian and Italian-American'],
}  

def get_merged_group(cultural_group):
    for k, v in merged_group_labels.items():
        if cultural_group in v:
            return k
    return cultural_group

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
    cultural_groups = [get_merged_group(cultural_groups[i]) for i in range(len(cultural_groups))]
    topics = reddit_data['topic']
    # take the cultural groups that have 10 or more members in them based on cc
    cc = Counter(cultural_groups)
    cc = {k: v for k, v in cc.items() if v >= 10}
    cultural_groups = [cultural_groups[i] if cultural_groups[i] in cc else "Other" for i in range(len(cultural_groups))]
    
    cluster_num = len(Counter(cultural_groups).items())
    # save {'id': i, 'contents': reddit_concat[i]} in jsonl format
    with open(save_path + "/reddit.jsonl", "w") as f:
        for i in range(len(reddit_concat)):
            obj = {'id': i, 'contents': reddit_concat[i], 'cultural_group': cultural_groups[i], 'topic': topics[i]}
            f.write(json.dumps(obj) + "\n")
        f.close()
    with open(save_path + f"/reddit-cluster_num={cluster_num}.txt", "w") as f:
        f.write(str(cluster_num))
        f.close()
    cultural_groups = tiktok_data['cultural group']
    cultural_groups = [get_merged_group(cultural_groups[i]) for i in range(len(cultural_groups))]
    topics = tiktok_data['topic']
    cc = Counter(cultural_groups)
    # take the cultural groups that have 10 or more members in them based on cc
    cc = {k: v for k, v in cc.items() if v >= 10}
    cultural_groups = [cultural_groups[i] if cultural_groups[i] in cc else "Other" for i in range(len(cultural_groups))]
    cluster_num = len(Counter(cultural_groups).items())
    with open(save_path + "/tiktok.jsonl", "w") as f:
        for i in range(len(tiktok_concat)):
            obj = {'id': i, 'contents': tiktok_concat[i], 'cultural_group': cultural_groups[i], 'topic': topics[i]}
            f.write(json.dumps(obj) + "\n")
        f.close()
    with open(save_path + f"/tiktok-cluster_num={cluster_num}.txt", "w") as f:
        f.write(str(cluster_num))
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

    
# UMAP, TSNE
# Kmeans, HAC, Top2Vec

# Top2Vec, TSNE - Kuleen
# UMAP, HAC - Abe
# Kmeans - Linda