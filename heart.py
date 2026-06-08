import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, render_template, redirect, url_for
import pickle


# Load and prepare the dataset
data = pd.read_csv(r"marco.csv")  # Fix for Windows file path
print(data.head())
data.rename(columns={"Heart Disease":"target"},inplace=True)

# Prepare features and target

X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'heart_attack_prediction_model.pkl')

# Create the Flask web application
app = Flask(__name__)  # Correct Flask initialization

# Load the trained model
model = joblib.load('heart_attack_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    feature_names = ['age', 'bp', 'cholesterol', 'max_hr', 'exercise_angina', 'st_depression', 'num_vessels', 'thallium']

    # Mapping for categorical variables
    thallium_mapping = {"normal": 3, "fixed_defect": 6, "reversible_defect": 7} 
    angina_mapping = {"yes": 1, "no": 0}
    try:
        # Get data from form and process it
        data = []
        for feature in feature_names:
            value = request.form.get(feature)
            if feature == 'thallium' and value in thallium_mapping:
                data.append(thallium_mapping[value])
            elif feature == 'exercise_angina' and value in angina_mapping:
                data.append(angina_mapping[value])
            else:
                data.append(float(value))  
        # Convert to numpy array
        input_data = np.array([data])
        print(f"Shape of X:{X.shape}")

        # Predict the outcome
        prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of class 1 (heart disease)
        prediction = 1 if prediction_proba > 0.5 else 0

        # Determine the output message
        output = 'has heart disease' if prediction == 1 else 'does not have heart disease'
        # Render the result in the predict page
        return render_template('predict.html', prediction_text= f'The person {output}')

    except ValueError as e:
        # Handle invalid input
        return render_template('predict.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":  # Correct way to run the Flask app
    app.run(debug=True)
