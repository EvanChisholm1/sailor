import numpy as np

class Index:
    def __init__(self):
        self.embeddings = []
        pass
    
    def search(self, query, k):
        pass

    def knn(self, target_embedding, k):
        pass

    def add_item(self):
        pass

    def save_file(self):
        pass

def calculate_similarity(a, b):
    dot_product = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    sim = dot_product / (normA * normB);
    return sim