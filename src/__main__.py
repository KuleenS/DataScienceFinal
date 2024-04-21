import argparse

import os

import tomllib

from typing import Dict, Any

import matplotlib.pyplot as plt

from src.clustering import *

from src.datasets import *

from src.dimension_reduction import *

def get_clustering(clustering_name: str, kwargs: Dict[str, Any]) -> ClusteringAlgorithm:

    name_to_cluster = {
        "hbdscan": HBDSCANCluster,
        "kmeans": KMeansCluster,
        "spectral": SpectralCluster
    }

    if clustering_name not in name_to_cluster:
        raise ValueError()
    
    return name_to_cluster[clustering_name](kwargs)


def get_dataset(dataset_name: str, kwargs: Dict[str, Any]) -> Dataset:

    name_to_dataset = {
        "aniso": AnisoSyntheticDataset,
        "blobs": BlobsSyntheticDataset,
        "circles": CirclesSyntheticDataset,
        "moons": MoonsSyntheticDataset,
        "nostructure": NoStructureSyntheticDataset,
        "socialnorms": SocialNormsDataset,
    }

    if dataset_name not in name_to_dataset:
        raise ValueError()
    
    if dataset_name != "socialnorms":
        dataset = name_to_dataset[dataset_name](**kwargs)

        return dataset.generate_dataset()
    else:
        dataset: Dataset = name_to_dataset[dataset_name]()
    
        return dataset.generate_dataset(kwargs)

def get_dimension_reducer(reducer_name: str, kwargs: Dict[str, Any]) -> DimensionReduction:

    name_to_reducer = {
        "diffusion": DiffusionMapReducer,
        "isomap" : IsomapReducer,
        "kpca": KernelPCAReducer,
        "lle" : LocalLinearEmbeddingReducer,
        "mds" : MultiDimensionalScalingReducer,
        "pca" : PCAReducer,
        "spectral" : SpectralReducer,
        "tsne" : TSNEReducer,
        "umap": UMAPReducer
    }

    if reducer_name not in name_to_reducer:
        raise ValueError()
    
    return name_to_reducer[reducer_name](kwargs)
    

def visualize(X, labels, name):
    plt.scatter(X[:, 0], X[:, 1], c=labels)

    plt.savefig(os.path.join("visualization", f"{name}.png"))

    plt.clf()
    

def main(args):

    config_path = args.config

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if "clustering" not in config:
        raise ValueError()

    if "datasets" not in config:
        raise ValueError()
    
    os.makedirs("visualization/")

    clustering_methods = config["clustering"]

    datasets = config["datasets"]

    if "socialnorms" in datasets and "dim_red" not in config:
        raise ValueError()

    for dataset in datasets:

        X,y = get_dataset(dataset, config.get(dataset, {}))

        for clustering_method in clustering_methods:

            clustering_algo = get_clustering(clustering_method, config.get(clustering_method, {}))

            labels = clustering_algo.cluster(X)

            if dataset == "socialnorms":
                
                dimension_reducers = config["dim_red"]

                for dimension_reducer in dimension_reducers:

                    reducer = get_dimension_reducer(dimension_reducer, config.get(clustering_method, {}))

                    reduced_X = reducer.reduce(X)

                    name = f"{dataset}_{clustering_method}_{dimension_reducer}"

                    visualize(reduced_X, labels, name)
            
            else:
                name = f"{dataset}_{clustering_method}"

                visualize(X, labels, name)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    main(args)
