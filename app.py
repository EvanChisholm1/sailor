from flask import Flask, request
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer
import numpy as np

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
    most_similar = {}
    highest_similarity = 0

    for e in embeddings:
        dot_product = np.dot(e['embedding'], embededQ)
        normA = np.linalg.norm(e['embedding'])
        normB = np.linalg.norm(embededQ)
        sim = dot_product / (normA * normB);

        if sim > highest_similarity:
            most_similar = e
            highest_similarity = sim
        
    print(most_similar)

    return "..."
