import os
from flask import Flask, render_template, request
from classifier import classifyImage
from flask import jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
# cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# app = Flask(__name__)


count = 0

hM = {"L": 0, "R": 0, "F": 0}


@app.route("/classify", methods=["POST"])
@cross_origin()
def classify():
    global count
    global hM
    result = "Collecting"
    if request.files["image"]:
        file = request.files["image"]
        tempResult = classifyImage(file)
        # print("Model classification: " + tempResult)
        print(tempResult, count, hM)
        if count == 20:
            if hM["L"] > 15:
                result = "L"
            elif hM["R"] > 15:
                result = "R"
            elif hM["F"] > 15:
                result = "F"
            # else:
            #     if hM["L"] > hM["R"] and hM["L"] > hM["F"]:
            #         result = "L"
            #     elif hM["R"] > hM["L"] and hM["R"] > hM["F"]:
            #         result = "R"
            #     else:
            #         result = "F"
            count = 0
            hM = {"L": 0, "R": 0, "F": 0}
        else:
            count += 1
            hM[tempResult] += 1
    return jsonify(finalResult=result, message="Success", statusCode=200), 200

if __name__ == '__main__':
    app.run(threaded=True, port=5000)