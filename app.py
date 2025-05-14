import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open("soil_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/preprocess", methods=["POST"])
def preprocess_data():
    try:
        moisture = float(request.form["moisture"])
        temp = float(request.form["temp"])
        absorb1 = float(request.form["absorb1"])
        absorb2 = float(request.form["absorb2"])
        pressure = float(request.form["pressure"])
        humidity = float(request.form["humidity"])
        carbonate = float(request.form["carbonate"])

        features = np.array([[moisture, temp, absorb1, absorb2, pressure, humidity, carbonate]])


        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"Error processing input: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
