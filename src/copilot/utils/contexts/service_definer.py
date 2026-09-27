import logging
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger(__name__)


class ClosestService(BaseModel):
    title: str
    description: str


class AsyncServiceFinder:
    """Finder of the closest service to the user query."""

    def __init__(
        self,
        services: List[str],
        service_vectors: np.ndarray,
        model_name: str = "johnnyboycurtis/ModernBERT-small",
    ) -> None:
        self.services = services
        self.service_vectors = service_vectors
        self.model_name = model_name

        # Загружаем модель один раз
        self.embedder = SentenceTransformer(model_name)

    @classmethod
    def create(
        cls,
        services_path: Path = Path(
            "./src/data/service_definer/services.pkl"
        ),
        vectors_path: Path = Path(
            "./src/data/service_definer/vectorized_services.pkl"
        ),
        model_name: str = "johnnyboycurtis/ModernBERT-small",
    ) -> "AsyncServiceFinder":

        services = cls._load_pickle(services_path)
        service_vectors = cls._load_pickle(vectors_path)

        service_vectors = np.asarray(service_vectors)

        if service_vectors.ndim == 3:
            service_vectors = service_vectors.squeeze(axis=1)

        return cls(
            services=services,
            service_vectors=service_vectors,
            model_name=model_name,
        )

    @staticmethod
    def vectorize_and_save_services(
        services: list[str],
        services_path: Path,
        vectors_path: Path,
        model_name: str = "johnnyboycurtis/ModernBERT-small",
        batch_size: int = 32,
    ) -> None:
        """
        Saves services and their embeddings to pickle files.
        """

        services_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        vectors_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = SentenceTransformer(model_name)

        vectors = model.encode(
            services,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        with open(services_path, "wb") as f:
            pickle.dump(services, f)

        with open(vectors_path, "wb") as f:
            pickle.dump(vectors, f)

        print(f"Saved services: {services_path}")
        print(f"Saved vectors: {vectors_path}")
        print(f"Services: {len(services)}")
        print(f"Vectors shape: {vectors.shape}")
    

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    def embed(self, text: str) -> np.ndarray:
        text = text.strip().replace("\n", " ")

        vector = self.embedder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vector

    def find_closest_service(
        self,
        query: str,
    ) -> ClosestService:

        logger.info("Query: %s", query)

        q_vec = self.embed(query).reshape(1, -1)

        similarities = cosine_similarity(
            q_vec,
            self.service_vectors,
        )

        idx = int(np.argmax(similarities[0]))

        raw = self.services[idx]

        title, description = [
            part.strip()
            for part in raw.split("|", 1)
        ]

        result = ClosestService(
            title=title,
            description=description,
        )

        logger.info("Result: %s", result)

        return result

    def find_closest_services(
        self,
        queries: List[str],
    ) -> List[ClosestService]:

        queries = [
            query.strip().replace("\n", " ")
            for query in queries
        ]

        # Один batch вместо N отдельных encode
        query_vectors = self.embedder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarities = cosine_similarity(
            query_vectors,
            self.service_vectors,
        )

        results = []

        for idx in np.argmax(similarities, axis=1):
            raw = self.services[int(idx)]

            title, description = [
                part.strip()
                for part in raw.split("|", 1)
            ]

            results.append(
                ClosestService(
                    title=title,
                    description=description,
                )
            )

        return results

service_name_finder = AsyncServiceFinder.create(
    services_path=Path('./src/data/service_definer/services.pkl'),
    vectors_path=Path('./src/data/service_definer/vectorized_services.pkl'),
    model_name="johnnyboycurtis/ModernBERT-small",
)
