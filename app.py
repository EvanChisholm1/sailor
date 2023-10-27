from flask import Flask, request
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from knn import KnnIndex

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
index = KnnIndex(model)
index.load_file('./db.json')

@app.route("/", methods=["GET", "POST"])
@cross_origin()
def main():
    if request.method == 'POST':
        print("post method")
        print(f"URL: {request.json['url']}\n")

        for sentence in getTextChunks(request.json['text']):
            index.add_item(request.json['url'], sentence)

        for e in embeddings:
            print(e['url'], "\n")

        return '{"message": "embedded %s"}' % request.json['url']
            

    return '{"message": "Hello world"}'


@app.get("/search")
@cross_origin()
def search():
    q = request.args.get('q')

    print("searched for:", q)

    most_similar = index.search(q, 10)

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
