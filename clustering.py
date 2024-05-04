from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from data_utils import read_cultural_bank
from sklearn.manifold import TSNE 
from tqdm import tqdm
import argparse
import umap
import pickle
from top2vec import Top2Vec
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

def run_clustering(args):
    embeddings_path = args.embeddings_path
    n_clusters = args.n_clusters
    algo = args.algo

    with open(embeddings_path, "rb") as f:
        corpus_embeddings = pickle.load(f)
        f.close()

    corpus, _, _ = read_cultural_bank(args.corpus)

    if algo == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters)
        save_path = embeddings_path.split(".")[0] + f"kmeans_clusters={n_clusters}.pkl"
        cluster_save_path = embeddings_path.split(".")[0] + f"kmeans_assignments={n_clusters}.npy"
        
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        cluster_centers = clustering_model.cluster_centers_
        with open(save_path, "wb") as f:
            pickle.dump(cluster_centers, f)
            f.close()
        with open(cluster_save_path, "wb") as f:
            np.save(f, cluster_assignment)
            f.close()

    elif algo == "agg":
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=args.dist_thres)
        save_path = embeddings_path.split(".")[0] + f"agg_clusters-dist_thres={args.dist_thres}.pkl"
        cluster_save_path = embeddings_path.split(".")[0] + f"agg_assignments-dist_thres={args.dist_thres}.npy"

        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in tqdm(enumerate(cluster_assignment), total=len(cluster_assignment), desc="Clustering"):
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
            clustered_sentences[cluster_id].append(corpus[sentence_id][0])

        count = [len(clustered_sentences[i]) for i in range(len(clustered_sentences))]
        print(Counter(count))

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


def umap_plot_clusters(data, labels, args):
    umap_args = {'n_neighbors': args.nn,
                'n_components': 2,
                'metric': 'cosine'}
    umap_model = umap.UMAP(**umap_args)
    transformed = umap_model.fit_transform(data)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels, cmap='Spectral', s=5)
    plt.title(f"UMAP projection of {args.algo} clusters, nn={args.nn}")
    plt.savefig(f"{args.algo}-cluster_plot-nn={args.nn}.png")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str)
    parser.add_argument("embeddings_path", type=str, help="The path to the embeddings")
    parser.add_argument("--cluster_assignments", type=str, default=None, help="The path to the cluster assignments, a list of labels")
    parser.add_argument("--n_clusters", type=int, default=None)
    parser.add_argument("--dist_thres", type=float, default=1.5)
    parser.add_argument("--nn", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--ppl", type=int, default=30, help="Perplexity for TSNE")
    parser.add_argument("--func", type=str, default="cluster", choices=["cluster", "eval", "plot"])
    parser.add_argument("--dim_reduce", type=str, default="umap", choices=["umap", "tsne"])
    parser.add_argument("--algo", type=str, default="agg", choices=["kmeans", "agg", "topic2vec"])
    args = parser.parse_args()
    if args.func == "cluster":
        run_clustering(args)
    elif args.func == "eval":
        pass
    elif args.func == "plot":
        run_plot(args)
        
    