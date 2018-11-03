"""Classe de objetos de reconhecimento de voz dado um conjunto de embeddings."""
from sklearn.neighbors import NearestNeighbors


class VoiceRecogniser:
    """Classe de encapsulamento do reconhecedor de pessoas.

    Args:
        threshold (float): distância máxima para reconhecimento.
    """

    def __init__(self, threshold=0.2):
        self._persons = None
        self.embeddings = None
        self.embedding_to_person = None
        self.threshold = threshold
        self._nn = NearestNeighbors(n_neighbors=1, n_jobs=-1, metric='cosine')

    @property
    def persons(self):
        return self._persons

    @persons.setter
    def persons(self, value: dict):
        """Setter de pessoas.

        Args:
            value (dict): dicionário de nomes de pessoas (keys) e seus embeddings (values)
        """
        self._persons = value
        if value:
            self.embeddings = [tuple(e) for e in self._persons.values()]
            self.embedding_to_person = {tuple(e): p for p, e in self._persons.items()}
            self._fit_nn()

    def _fit_nn(self):
        self._nn.fit(self.embeddings)

    def recognise(self, embedding, threshold=None):
        if not threshold:
            threshold = self.threshold

        distances, indices = self._nn.kneighbors([embedding], n_neighbors=1)
        distance = distances[0][0]
        index = indices[0][0]
        if distance <= threshold:
            emb = self.embeddings[index]
            return self.embedding_to_person[emb], distance
        return None, None
