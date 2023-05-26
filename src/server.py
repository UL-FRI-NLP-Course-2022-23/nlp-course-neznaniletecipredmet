from flask import Flask, request, jsonify
import os
import csv
from analysis import analyse_story

app = Flask(__name__)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/process", methods=["POST"])
def procrelationship_weightsess():
    # Get the text input from the request
    text_input = request.json["text"]

    # Run your Python script with the input and get the result
    analyse_story(text_input)

    return jsonify({"success": True})


# @app.route("/getFiles", methods=["POST"])
# def get_files():
#     folder_path = "../data/fairytales/stories"
#     file_names = os.listdir(folder_path)
#     return jsonify(file_names)


@app.route("/example", methods=["POST"])
def example():
    for name in ["characters", "relationships"]:
        to_read = "static/data/farytales_" + name + ".csv"
        to_write = "static/data/" + name + ".csv"
        with open(to_read, newline="", encoding='UTF8') as to_read_fp:
            with open(to_write, "w", newline="", encoding='UTF8') as to_write_fp:
                reader = csv.reader(to_read_fp)
                writer = csv.writer(to_write_fp)
                for row in reader:
                    writer.writerow(row)

    return jsonify({"success": True})


if __name__ == "__main__":
    app.run()
