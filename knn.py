import numpy as np

class KnnIndex:
    def __init__(self, model):
        self.embeddings = []
        self.model = model
        pass
    
    def search(self, query, k):
        e = self.model.encode(query)
        return self.knn(e, k)

    def knn(self, target, k):

        most_similar = [{'embedding': None, 'similarity': -float('inf')}] * k;

        for e in self.embeddings:
            sim = calculate_similarity(e['embedding'], target)

            if sim > most_similar[-1]['similarity']:
                print('greater similarity')
                most_similar.pop()
                most_similar.append({
                    'embedding': e,
                    'similarity': sim
                })
                # really slow but works for now
                most_similar = sorted(most_similar, key=lambda x:x['similarity'], reverse=True)

                # make sure the same page does not show up multiple times due to the fact that there are multiple chunks of each website in the embedding list
                urls = set()
                for embedding_check in most_similar:
                    if embedding_check['embedding'] == None: continue
                    if embedding_check['embedding']['url'] in urls:
                        most_similar.remove(embedding_check)
                        most_similar.append({'embedding': None, 'similarity': -float('inf')})
                    
                    urls.add(embedding_check['embedding']['url'])

        return most_similar

    def add_item(self, url, sentence, embedding = None):
        self.embeddings.append({
            'url': url,
            'sentence': sentence,
            'embedding': embedding if embedding != None else self.model.encode(sentence)
        })

    def save_file(self, path):
        pass

    def load_file(self, path):
        pass

def calculate_similarity(a, b):
    dot_product = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    sim = dot_product / (normA * normB);
    return sim