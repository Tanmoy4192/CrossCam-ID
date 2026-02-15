import numpy as np
from threading import Lock


class Tracker:
    """
    Matches embeddings to identities using cosine similarity.
    Known identities are stored in a dict.
    Unknown identities are stored and assigned person_X IDs.
    """

    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.known = {}          # {"name": embedding}
        self.unknowns = []       # [(id_name, embedding)]
        self.next_id = 1
        self.lock = Lock()       # thread safety (shared across cameras)

    def match(self, embedding):
        with self.lock:
            # 1. match known identities
            for name, emb in self.known.items():
                if self.cosine(embedding, emb) > self.threshold:
                    return name

            # 2. match previously seen unknowns
            for id_name, emb in self.unknowns:
                if self.cosine(embedding, emb) > self.threshold:
                    return id_name

            # 3. new person
            new_id = f"person_{self.next_id}"
            self.unknowns.append((new_id, embedding))
            self.next_id += 1
            return new_id

    def cosine(self, a, b):
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return dot / norm if norm != 0 else 0.0
