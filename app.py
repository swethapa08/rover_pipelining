from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

try:
    model = pickle.load(open("soil_model.pkl", "rb"))
except Exception as e:
    print(f"Error loading soil_model.pkl: {e}")
    raise

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
        ninhydrin = float(request.form["ninhydrin"])

        if carbonate not in [0, 1] or ninhydrin not in [0, 1]:
            return render_template("result.html", prediction="Error: Carbonate and Ninhydrin must be 0 or 1", carbonate=carbonate, ninhydrin=ninhydrin)

        features = np.array([[moisture, temp, absorb1, absorb2, pressure, humidity, carbonate, ninhydrin]])

        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=prediction, carbonate=int(carbonate), ninhydrin=int(ninhydrin))
    except ValueError as e:
        return render_template("result.html", prediction=f"Error: Invalid numerical input ({str(e)})", carbonate=0, ninhydrin=0)
    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}", carbonate=0, ninhydrin=0)

if __name__ == "__main__":
    app.run(debug=True)
