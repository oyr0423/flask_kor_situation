from flask import Flask
from flask_restful import Resource, Api, reqparse, abort
from flask import request

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
api = Api(app)


##
sentences = ["내가 새로 시작할 만한 취미가 있을까? ",
             "해야될 일이 있는데 자꾸 미루게 돼.",
             "나에게 어울릴 것 같은 취미",
             "오늘 저녁 뭐먹지?"
]

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

sentence_embeddings = model.encode(sentences)

print(len(sentence_embeddings[0]))

print (cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
))


similarity_score = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
).flatten()

print(similarity_score)

pd.DataFrame({"Sentence":sentences[1:],"Similarity_Score":similarity_score })

result_array = []
for i in range(0, 3):
    result_array.append({'hello' : float(similarity_score[i])})

print(result_array)

result_list = list(result_array)


todos = {
    1 : {"task":"Write Hello World Program", "summary":"write the code using python."},
    2 : {"task" : "Task2", "summary":"writing task2."},
    3 : {"task" : "Task3", "summary" : "this is task3."}
}

class HelloWorld(Resource):
    def get(self):
        # return {'hello': len(sentence_embeddings[0])}
        return result_list

class ToDoList(Resource):
    def get(self):
        return todos

class ToDo(Resource):
    def get(self, todo_id):
        return todos[todo_id]




@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    modi_sentence = str(sentence)
    encoded_sentence = model.encode(modi_sentence)
    similarity_score = cosine_similarity(
        [encoded_sentence],
        sentence_embeddings[:]
        ).flatten()

    result_array = []
    for i in range(0, 4):
        result_array.append({'hello' : float(similarity_score[i])})
        result_list = list(result_array)

    return result_list


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=True, host="0.0.0.0", port="9999")