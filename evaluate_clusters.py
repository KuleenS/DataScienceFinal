from data_utils import read_cultural_bank
from embed_data import embed_data
from clustering import HAC, kmeans_cluster, umap_plot_clusters
import torch
from sentence_transformers import SentenceTransformer

def evaluate_us_clusters(args):
    data, topics, cultural_groups = read_cultural_bank(args.path)
    us_data, us_topics = [], []
    ny_data, ny_topics = [], []
    ag_data, ag_topics = [], []
    for i in range(len(data)):
        if cultural_groups[i] == "Americans":
            us_data.append(data[i])
            us_topics.append(topics[i])
        elif cultural_groups[i] == "New Yorkers":
            ny_data.append(data[i])
            ny_topics.append(topics[i])
        elif cultural_groups[i] == "Argentinians":
            ag_data.append(data[i])
            ag_topics.append(topics[i])
    
    sbert = SentenceTransformer(args.model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    us_embeddings = embed_data(us_data, sbert, args.save_path)
    ny_embeddings = embed_data(ny_data, sbert, args.save_path)
    ag_embeddings = embed_data(ag_data, sbert, args.save_path)


    us_mean = torch.mean(us_embeddings, dim=0)
    ny_mean = torch.mean(ny_embeddings, dim=0)
    ag_mean = torch.mean(ag_embeddings, dim=0)
    # pairwise cosine sim
    us_ny_sim = torch.cosine_similarity(us_mean, ny_mean, dim=0)
    us_ag_sim = torch.cosine_similarity(us_mean, ag_mean, dim=0)
    ny_ag_sim = torch.cosine_similarity(ny_mean, ag_mean, dim=0)
    

    us_cluster_assignment = HAC(us_embeddings, n_clusters=args.n_clusters, dist_thres=args.dist_thres)
    ny_cluster_assignment = HAC(ny_embeddings, n_clusters=args.n_clusters, dist_thres=args.dist_thres)
    ag_cluster_assignment = HAC(ag_embeddings, n_clusters=args.n_clusters, dist_thres=args.dist_thres)

        
        
    