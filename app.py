import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, request, render_template, redirect, url_for


# Load and prepare the dataset
data = pd.read_csv(r"C:\heart\heart_attack_prediction_dataset.csv")  # Fix for Windows file path
print(data.head())

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
app = Flask(_name_)  # Correct Flask initialization

# Load the trained model
model = joblib.load('heart_attack_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Mapping for categorical variables
    thallium_mapping = {"normal": 3, "fixed_defect": 6, "reversible_defect": 7}

    try:
        # Get data from form and process it
        data = []
        for key, value in request.form.items():
            if key == 'thallium':
                data.append(thallium_mapping[value])  # Convert thallium to numerical value
            elif key == 'exercise_angina':
                data.append(1 if value == 'yes' else 0)  # Convert exercise_angina to binary
            else:
                data.append(float(value))  # Convert other values to float

        # Convert to numpy array
        input_data = np.array([data])

        # Predict the outcome
        prediction = model.predict(input_data)

        # Determine the output message
        output = 'Has Heart Disease' if prediction[0] == 1 else 'Does Not Have Heart Disease'

        # Render the result in the predict page
        return render_template('predict.html', prediction_text=f'The person {output}')

    except ValueError as e:
        # Handle invalid input
        return render_template('predict.html', prediction_text=f'Error: {str(e)}')

if _name_ == "_main_":  # Correct way to run the Flask app
    app.run(debug=True)