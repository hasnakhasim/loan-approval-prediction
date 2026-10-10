
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import traceback

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.pkl")


def load_model():
    """Load the trained machine learning model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run: python train_model.py"
        )

    return joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get applicant information from the HTML form
        applicant = {
            "Gender": request.form["Gender"],
            "Married": request.form["Married"],
            "Dependents": request.form["Dependents"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
            "Credit_History": int(request.form["Credit_History"]),
            "Property_Area": request.form["Property_Area"]
        }

        # 2. Validate numeric values
        numeric_fields = [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term"
        ]

        for field in numeric_fields:
            if applicant[field] < 0:
                raise ValueError(
                    f"{field.replace('_', ' ')} cannot be negative."
                )

        if applicant["Loan_Amount_Term"] <= 0:
            raise ValueError("Loan term must be greater than zero.")

        if applicant["Credit_History"] not in [0, 1]:
            raise ValueError("Credit history must be 0 or 1.")

        # 3. Convert applicant information into a DataFrame
        input_df = pd.DataFrame([applicant])

        # 4. Load the trained model
        model = load_model()

        # 5. Check that the form matches the model's features
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            missing = [col for col in expected if col not in input_df.columns]

            if missing:
                raise ValueError(
                    "Form fields do not match the trained model. "
                    f"Missing columns: {missing}. "
                    "Check train_model.py and retrain the model."
                )

            input_df = input_df[expected]

        # 6. Predict loan status
        prediction = model.predict(input_df)[0]

        # 7. Get the model's estimated probability
        probabilities = model.predict_proba(input_df)[0]

        # Works with the supplied Pipeline containing a classifier step
        classifier = model.named_steps["classifier"]
        classes = list(classifier.classes_)

        confidence = float(
            probabilities[classes.index(prediction)]
        ) * 100

        approved = str(prediction).strip().upper() == "Y"

        # 8. Display the result page
        return render_template(
            "result.html",
            approved=approved,
            prediction=(
                "Likely Approved" if approved
                else "Likely Rejected"
            ),
            confidence=confidence,
            applicant=applicant,
            error=None
        )

    except (ValueError, KeyError) as error:
        app.logger.exception("Invalid form data")
        return render_template(
            "result.html",
            error=str(error)
        ), 400

    except FileNotFoundError as error:
        app.logger.exception("Model file error")
        return render_template(
            "result.html",
            error=str(error)
        ), 500

    except Exception as error:
        print("\n========== PREDICTION ERROR ==========")
        traceback.print_exc()
        print("Error message:", str(error))
        print("======================================\n")

        return render_template(
            "result.html",
            error=(
                "Prediction failed: "
                + str(error)
                + ". Check the VS Code terminal for details."
            )
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
