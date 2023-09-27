from flask import Flask, request
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from knn import calculate_similarity

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def getTextChunks(text):
    words = text.split()

    sentences = []
    recent = []
    for word in words:
        recent.append(word)
        if len(recent) >= 200:
            sentence = " ".join(recent)
            sentences.append(sentence)
            recent = []
    
    return sentences

embeddings = []

@app.route("/", methods=["GET", "POST"])
@cross_origin()
def main():
    if request.method == 'POST':
        print("post method")
        print(f"URL: {request.json['url']}\n")

        for sentence in getTextChunks(request.json['text']):
            embeddings.append({
                'url': request.json['url'], 
                'sentence': sentence, 
                'embedding': model.encode(sentence)
            })

        for e in embeddings:
            print(e['url'], "\n")

        return '{"message": "embedded %s"}' % request.json['url']
            

    return '{"message": "Hello world"}'


@app.get("/search")
@cross_origin()
def search():
    query = request.args.get('q')

    print("searched for:", query)

    embededQ = model.encode(query)
    most_similar = [{'embedding': None, 'similarity': -float('inf')}] * 10;

    for e in embeddings:
        sim = calculate_similarity(e['embedding'], embededQ)

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

                
        
    for i in most_similar:
        if not i['embedding'] == None:
            print(i['embedding']['url'], f"\n {i['embedding']['sentence']} \n\n")
        
    # format of API is rather weird but is like that to support my pre-existing frontend for WTFDIST
    out = [{
        'id': e['embedding']['url'],
        'title': e['embedding']['url'],
        'content': e['embedding']['sentence'],
        'link': e['embedding']['url'],
    } for e in most_similar if not e['embedding'] == None]

    return json.dumps(out)
