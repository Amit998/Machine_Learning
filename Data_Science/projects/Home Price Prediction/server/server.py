from flask import Flask,request,jsonify
import util

app=Flask(__name__)


@app.route('/hello')
def hello():
    return "Hi"


@app.route('/predict_home_price',methods=['POST'])
def predict_home_price():
    location=str(request.form['location'])
    total_sqft=float(request.form['total_sqft'])
    bhk=float(request.form['bhk'])
    bath=float(request.form['bath'])

    response=jsonify({
        'estimated_price':util.get_estimated_price(location,total_sqft,bhk,bath)
    })
    return response


@app.route('/get_location_names')
def get_location_names():
    response=jsonify({
        'locations':util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response


if __name__ == "__main__":
    
    print("Staring Python Flask Server for Home Prediction...")
    app.run()