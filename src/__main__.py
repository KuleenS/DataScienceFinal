import argparse

import tomllib

from src.clustering.clustering_algo import ClusteringAlgorithm
from src.clustering.hbdscan_cluster import HBDSCANCluster
from src.clustering.kmeans_cluster import KMeansCluster
from src.clustering.spectral_cluster import SpectralCluster


from src.datasets.aniso import AnisoSyntheticDataset
from src.datasets.blobs import BlobsSyntheticDataset
from src.datasets.circles import CirclesSyntheticDataset
from src.datasets.dataset import Dataset
from src.datasets.moons import MoonsSyntheticDataset
from src.datasets.no_structure import NoStructureSyntheticDataset
from src.datasets.social_norms import SocialNormsDataset


def get_clustering(clustering_name, kwargs) -> ClusteringAlgorithm:

    name_to_cluster = {
        "hbdscan": HBDSCANCluster,
        "kmeans": KMeansCluster,
        "spectral": SpectralCluster
    }

    if clustering_name not in name_to_cluster:
        raise ValueError()
    
    return name_to_cluster[clustering_name](kwargs)


def get_dataset(dataset_name, kwargs) -> Dataset:

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



def main(args):

    config_path = args.config

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if "clustering" not in config:
        raise ValueError()

    if "datasets" not in config:
        raise ValueError()

    clustering_methods = config["clustering"]

    datasets = config["datasets"]

    for dataset in datasets:

        X,y = get_dataset(dataset, config.get(dataset, {}))

        for clustering_method in clustering_methods:

            clustering_algo = get_clustering(clustering_method, config.get(clustering_method, {}))

            labels = clustering_algo.cluster(X)

            # ima be 100 with you abe, idk what are we doing here 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    main(args)
