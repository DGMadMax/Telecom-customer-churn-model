# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\IBMR\Desktop\Jupyter Notebook\model.sav" 
model = pickle.load(open(model_path, "rb"))

df_1 = pd.read_csv(r"C:\Users\IBMR\Downloads\Churn_Model.csv")

@app.route("/")
def loadPage():
    return render_template("churn.html", query="")

@app.route("/", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        input_data = [
            request.form["query1"], request.form["query2"], request.form["query3"], request.form["query4"],
            request.form["query5"], request.form["query6"], request.form["query7"], request.form["query8"],
            request.form["query9"], request.form["query10"], request.form["query11"], request.form["query12"],
            request.form["query13"], request.form["query14"], request.form["query15"], request.form["query16"],
            request.form["query17"], request.form["query18"], request.form["query19"]
        ]

        # Convert numeric inputs to appropriate types
        numeric_indices = [0,1, 2, 18]  
        for i in numeric_indices:
            input_data[i] = float(input_data[i])


        # Create a DataFrame for input
        columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents', 
                   'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                   'PaperlessBilling', 'PaymentMethod', 'tenure']
        
        new_df = pd.DataFrame([input_data], columns=columns)

        # Ensure tenure is categorized into bins
        labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
        new_df["tenure_group"] = pd.cut(new_df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

        # Drop tenure column (as done in training)
        new_df.drop(columns=["tenure"], axis=1, inplace=True)

        # Apply one-hot encoding (match training feature set)
        new_df_dummies = pd.get_dummies(new_df)

        # Ensure new input matches trained model's feature set
        expected_features = model.get_booster().feature_names
        new_df_dummies = new_df_dummies.reindex(columns=expected_features, fill_value=0)

        # Make predictions
        prediction = model.predict(new_df_dummies)
        probability = model.predict_proba(new_df_dummies)[:, 1][0]

        # Format output
        if prediction == 1:
            result = "This Customer is likely to Churn!"
        else:
            result = "This Customer is likely to Continue!"
        
        confidence = f"Confidence: {probability * 100:.2f}%"

        return render_template("churn.html", output1=result, output2=confidence, **request.form)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
