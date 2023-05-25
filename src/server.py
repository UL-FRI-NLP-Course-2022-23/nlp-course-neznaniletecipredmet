from flask import Flask, request, jsonify
import os
from analyse import analyse_story

app = Flask(__name__)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/process", methods=["POST"])
def process():
    # Get the text input from the request
    text_input = request.json["text"]

    # Run your Python script with the input and get the result
    analyse_story(text_input)

    # Save the result to a CSV file
    # with open("static/data/characters.csv", "w", newline="", encoding='UTF8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for character in result:
    #         writer.writerow([character, 0.5, 0.5])
    #
    # with open("static/data/relationships.csv", "w", newline="", encoding='UTF8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for character in result:
    #         for character2 in result:
    #             if character != character2:
    #                 writer.writerow([character, character2, "", 0.5])

    return jsonify({"success": True})


@app.route("/getFiles")
def get_files():
    folder_path = "../data/fairytales/stories"
    file_names = os.listdir(folder_path)
    return jsonify(file_names)


if __name__ == "__main__":
    app.run()
