import json
import numpy as np

class KnnIndex:
    def __init__(self, model):
        self.embeddings = []
        self.model = model
        pass
    
    def search(self, query, k, duplicates=False):
        e = self.model.encode(query)
        return self.knn(e, k, duplicates)

    def knn(self, target, k, duplicates=False):
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

                if duplicates: continue

                # make sure the same page does not show up multiple times due to the fact that there are multiple chunks of each website in the embedding list
                urls = set()
                new_similar = []
                for embedding_check in most_similar:
                    if embedding_check['embedding'] == None: continue
                    if embedding_check['embedding']['url'] not in urls:
                        new_similar.append(embedding_check)
                        urls.add(embedding_check['embedding']['url'])
                
                while len(new_similar) < k:
                    most_similar.append({'embedding': None, 'similarity': -float('inf')})


        return most_similar

    def add_item(self, url, sentence, embedding = None):
        self.embeddings.append({
            'url': url,
            'sentence': sentence,
            'embedding': embedding if embedding != None else self.model.encode(sentence),
        })
    
        self.save_file("./db.json")

    # for now we just store a json file, may have to switch to something else at somepoint if it becomes slow
    def save_file(self, path):
        json_dict = {
            'embeddings': []
        }

        for e in self.embeddings:
            serializable = {
                'url': e['url'],
                'sentence': e['sentence'],
                'embedding': e['embedding'].tolist()
            }

            json_dict['embeddings'].append(serializable)
        
        with open(path, "w") as db_file:
            db_file.write(json.dumps(json_dict))

    def load_file(self, path):
        print('loading db...')
        with open(path, "r") as db_file:
            text = db_file.read()
            json_dict = json.loads(text)
            for item in json_dict['embeddings']:
                item['embedding'] = np.asarray(item['embedding'])
            
            self.embeddings = json_dict['embeddings']
        
        print('print db loaded!')
        


def calculate_similarity(a, b):
    dot_product = np.dot(a, b)
    normA = np.linalg.norm(a)
    normB = np.linalg.norm(b)
    sim = dot_product / (normA * normB);
    return sim