from flask import Flask, request, render_template
import pickle
import pandas as pd

application = Flask(__name__)
app = application

# Load Ridge model and StandardScaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict_datapoint():

    Temperature = float(request.form['Temperature'])
    RH = float(request.form['RH'])
    Ws = float(request.form['Ws'])
    Rain = float(request.form['Rain'])
    FFMC = float(request.form['FFMC'])
    DMC = float(request.form['DMC'])
    ISI = float(request.form['ISI'])
    Classes = float(request.form['Classes'])
    Region = float(request.form['Region'])

    input_data = pd.DataFrame(
        [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
        columns=[
            'Temperature',
            'RH',
            'Ws',
            'Rain',
            'FFMC',
            'DMC',
            'ISI',
            'Classes',
            'Region'
        ]
    )

    # Scale the input
    scaled_data = standard_scaler.transform(input_data)

    # Make prediction
    prediction = ridge_model.predict(scaled_data)

    result = prediction[0]

    return render_template('home.html', result=result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)