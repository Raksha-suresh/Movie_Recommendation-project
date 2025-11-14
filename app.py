from flask import Flask, render_template, request
from recommender import Recommender

app = Flask(__name__)
r = Recommender()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form["title"]
    results = r.recommend(title)
    return render_template("results.html", query=title, results=results)

if __name__ == "__main__":
    app.run(debug=True)
