###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import json
import faiss
import numpy as np
from numpy.typing import NDArray
from typing import Any, Self


class FaissDataRetriever:
    def __init__(self, data: list[Any], embeddings: NDArray) -> None:
        """
        Args:
            data (list[Any]): a list of data samples to be retrieved from.
            embeddings (NDArray): embedding vectors corresponding to `data`.
                Must have shape `[B, d]`, where `B` is the number of data samples in database.
        """
        assert len(data) == len(embeddings)
        assert embeddings.ndim == 2
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlat(dim)
        self.faiss_index.add(embeddings)
        self.data = data

    @classmethod
    def from_files(cls, data_path: str, emb_path: str, data_format: str = "json") -> Self:
        """
        Args:
            data_path (str): path to data file for retrieval. Data must be indexable, e.g., a list.
            emb_path (str): path to npy file containing the embedding vectors for 'data_path'.
            data_format (str): data file format for 'data_path'. Default is json.
        """
        # Load data file
        match data_format:
            case "json":
                with open(data_path) as f:
                    data = [json.loads(line) for line in f]
            case _:
                raise NotImplementedError

        # Load embeddings
        emb = np.load(emb_path)

        return cls(data, emb)

    def search_similar(self, query: NDArray, k: int) -> tuple[list[list[float]], list[list[int]], list[list[Any]]]:
        """
        Args:
            query (NDArray): query embedding vectors. Must have shape `[N, d]`.
            k (int): number of similar data samples to retrieve, as in top-k search.

        Returns:
            distances (NDArray[np.float]): embedding distances of retrieved data samples to the query vectors.
            indices (NDArray[np.int]): indices of retrieved data samples.
            similar_data (list[list[Any]]): retrieved data samples.
        """
        assert query.ndim == 2

        D, I = self.faiss_index.search(query, k)
        similar = []
        for row in I:
            similar.append([self.data[i] for i in row])
        return D.tolist(), I.tolist(), similar
