from flask import Flask, request
from flask_cors import CORS, cross_origin
from sentence_transformers import SentenceTransformer

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


@app.route("/", methods=["GET", "POST"])
@cross_origin()
def main():
    if request.method == 'POST':
        print(f"URL: {request.json['url']}\n")

        for sentence in getTextChunks(request.json['text']):
            print(sentence, model.encode(sentence), '\n')
            print()
        # print(model.encode(request.json['text']))
        print("post method")

    return '{"message": "Hello world"}'

