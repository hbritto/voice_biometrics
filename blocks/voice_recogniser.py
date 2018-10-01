"""Classe de objetos de reconhecimento de voz dado um conjunto de embeddings."""
from sklearn.neighbors import NearestNeighbors


class VoiceRecogniser:
    def __init__(self):
        self.persons = None
        self.embedding_to_person = None
        self._nn = NearestNeighbors(n_neighbors=1, n_jobs=-1, metric='cosine')

    @property
    def persons(self):
        return self.persons

    @persons.setter
    def persons(self, value: dict):
        """Setter de pessoas.

        Args:
            value (dict): dicionário de pessoas (key) e seus embeddings (value)
        """
        self.persons = value
        self.embedding_to_person = {tuple(e): p for p, e in self.persons.items()}
        self._fit_nn(list(self.persons.values()))

    def _fit_nn(self, all_embeddings):
        self._nn.fit(all_embeddings)
