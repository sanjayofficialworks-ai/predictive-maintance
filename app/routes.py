from flask import Blueprint, request, jsonify,render_template

from app.inference import predict_failure


main = Blueprint("main", __name__)


@main.route("/", methods=["GET"])
def home():

    return render_template("index.html")


@main.route("/about", methods=["GET"])
def about():

    return render_template("about.html")


@main.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    result = predict_failure(data)

    return jsonify(result)