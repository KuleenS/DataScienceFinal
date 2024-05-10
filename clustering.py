from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from data_utils import read_cultural_bank
from sklearn.manifold import TSNE 
from tqdm import tqdm
import argparse
import umap
import pickle
from top2vec import Top2Vec
from sentence_transformers import SentenceTransformer
from embed_data import embed_data
import torch
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

def kmeans_cluster(corpus_embeddings, n_clusters):
    clustering_model = KMeans(n_clusters=n_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    cluster_centers = clustering_model.cluster_centers_
    return cluster_centers, cluster_assignment

def HAC(corpus_embeddings, n_clusters=None, dist_thres=None):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=dist_thres)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    return cluster_assignment
    

def run_clustering(args):
    embeddings_path = args.embeddings_path
    n_clusters = args.n_clusters
    algo = args.algo

    with open(embeddings_path, "rb") as f:
        corpus_embeddings = pickle.load(f)
        f.close()

    corpus, _, _ = read_cultural_bank(args.corpus)

    if algo == "kmeans":
        save_path = embeddings_path.split(".")[0] + f"kmeans_clusters={n_clusters}.pkl"
        cluster_save_path = embeddings_path.split(".")[0] + f"kmeans_assignments={n_clusters}.npy"
        cluster_centers, cluster_assignment = kmeans_cluster(corpus_embeddings, n_clusters)
        
        with open(save_path, "wb") as f:
            pickle.dump(cluster_centers, f)
            f.close()
        with open(cluster_save_path, "wb") as f:
            np.save(f, cluster_assignment)
            f.close()

    elif algo == "agg":
        save_path = embeddings_path.split(".")[0] + f"agg_clusters-dist_thres={args.dist_thres}.pkl"
        cluster_save_path = embeddings_path.split(".")[0] + f"agg_assignments-dist_thres={args.dist_thres}.npy"
        cluster_assignment = HAC(corpus_embeddings, n_clusters=n_clusters, dist_thres=args.dist_thres)
        clustered_sentences = {}
        for sentence_id, cluster_id in tqdm(enumerate(cluster_assignment), total=len(cluster_assignment), desc="Clustering"):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
            clustered_sentences[cluster_id].append(corpus[sentence_id][0])

        count = [len(clustered_sentences[i]) for i in range(len(clustered_sentences))]
        types = Counter(cluster_assignment)
        print(len(types.keys()))

        with open(save_path, "wb") as f:
            pickle.dump(clustered_sentences, f)
            f.close()
        
        with open(cluster_save_path, "wb") as f:
            np.save(f, cluster_assignment)
            f.close()

    elif algo == "topic2vec":
        model = Top2Vec(corpus, embedding_model='universal-sentence-encoder-large')
        model_save_path = embeddings_path.split(".")[0] + "-top2vec_model"
        model.save(model_save_path)
        assigned_topics, _, _, _ = model.get_documents_topics(list(range(len(corpus))))
        assignment_save_path = embeddings_path.split(".")[0] + "-top2vec_assignments.npy"
        np.save(assignment_save_path, assigned_topics)


def umap_plot_clusters(data, labels, args, plot_tag):
    umap_args = {'n_neighbors': args.nn,
                'n_components': 2,
                'metric': 'cosine'}
    umap_model = umap.UMAP(**umap_args)
    transformed = umap_model.fit_transform(data)
    fig, ax = plt.subplots()
    if type(labels[0]) == str:
        ax.scatter(transformed[:, 0], transformed[:, 1], cmap='Spectral', s=5)
    else:
        ax.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='Spectral', s=5)
    # for i, txt in enumerate(labels):
    #     ax.annotate(txt, (transformed[:, 0][i], transformed[:, 1][i]), fontsize=3)
    plt.title(f"{plot_tag}, UMAP projection of {args.algo} clusters, nn={args.nn}")
    plt.savefig(f"{plot_tag}-{args.algo}-cluster_plot-nn={args.nn}.png")

def tsne_plot_clusters(data, labels, args):
    tsne_args = {'n_components': 2,
                'metric': 'cosine',
                'perplexity': args.ppl}
    tsne_model = TSNE(**tsne_args)
    transformed = tsne_model.fit_transform(np.array(data))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='Spectral', s=5)
    plt.title(f"TSNE projection of {args.algo} clusters, perplexity={tsne_args['perplexity']}")
    plt.savefig(f"{args.algo}-cluster_plot-ppl={args.ppl}.png")

def run_plot(args):
    with open(args.embeddings_path, "rb") as f:
        corpus_embeddings = pickle.load(f)
        f.close()
    cluster_assignments = np.load(args.cluster_assignments)
    if args.dim_reduce == "umap":
        umap_plot_clusters(corpus_embeddings, cluster_assignments, args)
    elif args.dim_reduce == "tsne":
        tsne_plot_clusters(corpus_embeddings, cluster_assignments, args)
    else:
        raise ValueError("Invalid dimensionality reduction method")

def eval_us_clusters(args):
    data, topics, cultural_groups = read_cultural_bank(args.corpus)
    us_data, us_topics = [], []
    ny_data, ny_topics = [], []
    ar_data, ar_topics = [], []
    for i in range(len(data)):
        if cultural_groups[i] == "Americans":
            us_data.append(data[i])
            us_topics.append(topics[i])
        elif cultural_groups[i] == "New Yorkers":
            ny_data.append(data[i])
            ny_topics.append(topics[i])
        elif cultural_groups[i] == "Argentinians":
            ar_data.append(data[i])
            ar_topics.append(topics[i])
    
    sbert = SentenceTransformer(args.model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    us_embeddings = embed_data(us_data, sbert, args.save_path)
    ny_embeddings = embed_data(ny_data, sbert, args.save_path)
    ar_embeddings = embed_data(ar_data, sbert, args.save_path)

    assert len(us_embeddings) == len(us_data)
    assert len(ny_embeddings) == len(ny_data)
    assert len(ar_embeddings) == len(ar_data)

    us_mean = np.mean(us_embeddings, axis=0)
    ny_mean = np.mean(ny_embeddings, axis=0)
    ar_mean = np.mean(ar_embeddings, axis=0)
    # breakpoint()

    us_mean_pt = torch.tensor(us_mean)
    ny_mean_pt = torch.tensor(ny_mean)
    ar_mean_pt = torch.tensor(ar_mean)

    # plot
    # umap_plot_clusters(cat, ["US", "NY", "AR"], args, plot_tag="mean vectors")
    
    # pairwise cosine sim
    us_ny_sim = torch.cosine_similarity(us_mean_pt, ny_mean_pt, dim=-1)
    us_ar_sim = torch.cosine_similarity(us_mean_pt, ar_mean_pt, dim=-1)
    ny_ar_sim = torch.cosine_similarity(ny_mean_pt, ar_mean_pt, dim=-1)
    print(f"US-NY similarity: {us_ny_sim}")
    print(f"US-ar similarity: {us_ar_sim}")
    print(f"NY-ar similarity: {ny_ar_sim}")

    # evaluate sub-clusters
    us_cluster_assignment = HAC(us_embeddings, n_clusters=args.n_clusters, dist_thres=args.dist_thres)
    ny_cluster_assignment = HAC(ny_embeddings, n_clusters=args.n_clusters, dist_thres=args.dist_thres)
    ar_cluster_assignment = HAC(ar_embeddings, n_clusters=args.n_clusters, dist_thres=args.dist_thres)

    umap_plot_clusters(us_embeddings, us_cluster_assignment, args, plot_tag="US subclusters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str)
    parser.add_argument("embeddings_path", type=str, help="The path to the embeddings")
    parser.add_argument("--save_path", type=str, default=None, help="The path to save the embeddings")
    parser.add_argument("--model_name", type=str, help="SentenceTransformer model name", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--cluster_assignments", type=str, default=None, help="The path to the cluster assignments, a list of labels")
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--dist_thres", type=float, default=None)
    parser.add_argument("--nn", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--ppl", type=int, default=30, help="Perplexity for TSNE")
    parser.add_argument("--func", type=str, default="cluster", choices=["cluster", "eval", "plot"])
    parser.add_argument("--dim_reduce", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--algo", type=str, default="agg", choices=["kmeans", "agg", "top2vec"])
    args = parser.parse_args()
    if args.func == "cluster":
        run_clustering(args)
    elif args.func == "eval":
        eval_us_clusters(args)
    elif args.func == "plot":
        run_plot(args)
        
    