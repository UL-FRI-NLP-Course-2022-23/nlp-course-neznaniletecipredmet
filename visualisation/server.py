from flask import Flask, request, jsonify
import csv
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
    result = analyse_story(text_input)

    # Save the result to a CSV file
    with open("result.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result)

    return jsonify({"success": True})

if __name__ == "__main__":
    app.run()
