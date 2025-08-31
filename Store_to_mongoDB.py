from pymongo import MongoClient
import pandas as pd

traning_set = pd.read_csv(r"C:/Users/TARANJEET SINGH/OneDrive/Desktop/CODING/VS Code (programms)/py/PROJECTS/Heart disease prediction/traning_dataSet.csv")
weightbias = pd.read_csv(r"C:/Users/TARANJEET SINGH/OneDrive/Desktop/CODING/VS Code (programms)/py/PROJECTS/Heart disease prediction/weight_bias.csv")
weight = weightbias["weight"].astype(float).tolist()
bias = weightbias["bias"].iloc[0]
URL="MongoDB_URL"
Client = MongoClient(URL)
db = Client["MedicalData"]
db["weight_bias"].drop()
db["traning_dataset"].drop()
db["max_values"].drop()

'''                                             storing traning dataSet                                                            '''
traning_data_dict = traning_set.to_dict(orient="records")
db["traning_dataset"].insert_many(traning_data_dict)

'''                                             Storing weight and bias                                                            '''
db["weight_bias"].insert_one({"weight":weight , "bias" : bias})

'''                                             Storing max values                                                                  '''
max_values = []
col = traning_set.columns
for i in range(len(col)):
    max_values.append(int(traning_set[col[i]].max()))
db["max_values"].insert_one({"max":max_values})    