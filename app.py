from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("car_price_model.pkl")

# Load the feature columns used during training
model_columns = joblib.load("model_columns.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        # Get data from the form
        car_age = int(request.form["car_age"])
        km_driven = int(request.form["km_driven"])
        fuel = request.form["fuel"]
        seller_type = request.form["seller_type"]
        transmission = request.form["transmission"]
        owner = request.form["owner"]

        # Create a DataFrame
        new_car = pd.DataFrame({
            "car_age": [car_age],
            "km_driven": [km_driven],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "owner": [owner]
        })

        # One-hot encoding
        new_car_encoded = pd.get_dummies(
            new_car,
            drop_first=True
        )

        # Make sure columns are exactly the same as training data
        new_car_encoded = new_car_encoded.reindex(
            columns=model_columns,
            fill_value=False
        )

        # Predict price
        predicted_price = model.predict(new_car_encoded)

        prediction = round(predicted_price[0], 2)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)