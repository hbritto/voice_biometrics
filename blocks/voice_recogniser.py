"""Classe de objetos de reconhecimento de voz dado um conjunto de embeddings."""
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor


class VoiceRecogniser:
    def __init__(self, threshold=0.2):
        self._persons = None
        self.embeddings = None
        self.embedding_to_person = None
        self.threshold = threshold
        self._nn = KNeighborsRegressor(n_neighbors=1, n_jobs=-1, metric='cosine')

    @property
    def persons(self):
        return self._persons

    @persons.setter
    def persons(self, value: dict):
        """Setter de pessoas.

        Args:
            value (dict): dicionário de pessoas (key) e seus embeddings (value)
        """
        self._persons = value
        if value:
            self.embeddings = [tuple(e) for e in self._persons.values()]
            self.embedding_to_person = {tuple(e): p for p, e in self._persons.items()}
            self._fit_nn()

    def _fit_nn(self):
        self._nn.fit(*zip(self.embeddings, list(self.persons.keys())))

    def recognise(self, embedding, threshold=None):
        if not threshold:
            threshold = self.threshold

        distances, indices = self._nn.kneighbors([embedding], n_neighbors=5)
        print(distances)
        distances = distances[0]
        print(indices)
        indices = indices[0]
        print(distances)
        print(indices)
        retn = []
        for distance, index in zip(distances, indices):
            if distance <= threshold:
                emb = self.embeddings[index]
                retn.append(self.embedding_to_person[emb])

        return retn
