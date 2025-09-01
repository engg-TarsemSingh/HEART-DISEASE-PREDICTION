from flask import Flask,request,jsonify
from pymongo import MongoClient
from Logistical_Regression import ApiPredict
import pandas as pd
from flask_cors import CORS

URL="MongoDb_URL"

client = MongoClient(URL)
db = client['MedicalData']

'''                                                 getting weight and bias values                                          '''
Weight_Bias = db["weight_bias"]
temp = list(Weight_Bias.find())
df = pd.DataFrame(temp)
weight = df["weight"].iloc[0]
bias = df["bias"].iloc[0]

'''                                                     getting maximum values                                               '''
max_value = db["max_values"]
temp = list(max_value.find())
df1 = pd.DataFrame(temp)
maxi = df1["max"].iloc[0]

app = Flask(__name__)
CORS(app)

@app.route("/predict",methods=["POST"])
def predict():
    data = request.json
    lis = []
    birth = pd.to_datetime(data["DOB"])
    now = pd.to_datetime("today")
    lis.append(int((now-birth)/ pd.Timedelta(days=365)))
    lis.append(int(data["Gender"]))
    lis.append(int(data["Chest_pain"]))
    lis.append(int(data["Blood_pressure"]))
    lis.append(int(data["Cholesterol"]))
    lis.append(int(data["FPS"]))
    lis.append(int(data["EKG"]))
    lis.append(int(data["Max_HR"]))
    lis.append(int(data["Exercise_angina"]))
    lis.append(float(data["ST_depression"]))
    lis.append(int(data["Slope_of_ST"]))
    lis.append(int(data["Number_of_vessels_fluro"]))
    lis.append(int(data["Thallium"]))
    
    predected = ApiPredict(lis,weight,bias,maxi)
    return jsonify({"predected" : predected})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

