from flask import Flask, render_template, request
from ml_pipeline import process_headline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        headline = request.form["headline"]
        paraphrased, sentiment, score = process_headline(headline)
        return render_template("index.html",
                               original=headline,
                               paraphrased=paraphrased,
                               sentiment=sentiment,
                               score=round(score, 2))
    return render_template("index.html", original=None)

if __name__ == "__main__":
    app.run(debug=True)
