from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest Classifier model
filename = './stacked_ensemble_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('./main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data and convert categorical fields to integers
        age = int(request.form['age'])
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form.get('slope'))
        ca = int(request.form['ca'])
        #thal = int(request.form['thal'])

        # Prepare data for prediction
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]])
        my_prediction = model.predict(data)

        # Render result with prediction
        return render_template('./result.html', prediction=my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
