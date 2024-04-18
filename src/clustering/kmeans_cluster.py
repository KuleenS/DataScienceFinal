from typing import Any, Dict, List

from sklearn.cluster import KMeans

from src.clustering.clustering_algo import ClusteringAlgorithm

class KMeansCluster(ClusteringAlgorithm):

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        super().__init__(kwargs)

        self.clustering = KMeans(**self.kwargs)

    
    def cluster(self, points: List[List[float]]) -> List[int]:
        return self.clustering.fit(points).labels_