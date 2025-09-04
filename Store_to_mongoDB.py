from pymongo import MongoClient
import pandas as pd

training_set = pd.read_csv(r"model_training/training_dataSet.csv")
weightbias = pd.read_csv(r"model_training/weight_bias.csv")
weight = weightbias["weight"].astype(float).tolist()
bias = weightbias["bias"].iloc[0]
URL="MongoDB_URL"
Client = MongoClient(URL)
db = Client["MedicalData"]
db["weight_bias"].drop()
db["training_dataset"].drop()
db["max_values"].drop()

'''                                             storing training dataSet                                                            '''
training_data_dict = training_set.to_dict(orient="records")
db["training_dataset"].insert_many(training_data_dict)

'''                                             Storing weight and bias                                                            '''
db["weight_bias"].insert_one({"weight":weight , "bias" : bias})

'''                                             Storing max values                                                                  '''
max_values = []
col = training_set.columns
for i in range(len(col)):
    max_values.append(int(training_set[col[i]].max()))

db["max_values"].insert_one({"max":max_values})    
