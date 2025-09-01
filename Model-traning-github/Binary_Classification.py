
'''                                     NOTE THE DATASET IS TAKEN FROM KAGGLE                       '''

import numpy as np
import pandas as pd


'''                     reading the dataSet and performing preprocessing                                         '''
df = pd.read_csv("Heart_Disease_Prediction.csv")
df["Heart Disease"] = (df["Heart Disease"]=="Presence").astype(int)


'''                           dividing the DataSet into two parts                                                '''
traning_dataSet = df[:150].copy()
testing_dataSet = df[150:].copy()
traning_dataSet.to_csv(r"traning_dataSet.csv",index=False)
testing_dataSet.to_csv(r"testing_dataSet.csv",index=False)


'''                                    Feature scalling                                                           '''
max_values = []
col = df.columns.tolist()
for i in range(df.shape[1]-1):
    maxi = df[col[i]].max()
    max_values.append(maxi)
    traning_dataSet[col[i]] = traning_dataSet[col[i]]/maxi


'''                           dividing traning set into feature and output                                         '''
x = traning_dataSet.iloc[:,:-1]
y = traning_dataSet.iloc[:,-1]


'''                                       weight bias initialize                                                   '''
def initialize_weightandbias(weights,biases):
    init = pd.DataFrame({ "weight" : weights ,
                           "bias"  : biases })
    init.to_csv(r"weight_bias.csv",index=False)


'''                                    Prediction function                                                         '''
def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict_yi(xi,weight,bias):
    z = np.dot(xi,weight)+bias
    return sigmoid(z)

def predict(x,weight,bias):
    result = []
    for i in range(x.shape[0]):
        y_hat = predict_yi(x.iloc[i],weight,bias)
        result.append(y_hat)
    return result


'''                                        error finction                                                          '''
def error(y_hati,yi):
    if yi==1 :
        return -np.log(y_hati)
    else :
        return -np.log(1-y_hati)

def total_error(y_hat,y):
    error_ = 0
    for i in range(len(y)):
        error_ += error(y_hat[i],y[i])
    return error_


'''                                          GRADIENT DESCEND                                                      '''
def gradientDescend(x,y,weight,bias,alpha,iteration):
    weight = np.array(weight)
    prediction_list = []
    error_list = []
    for kk in range(iteration):
        y_hat = predict(x,weight,bias)
        
        prediction_list.append(y_hat)
        error_list.append(total_error(y_hat,y))
        
        weight_number = [0]*len(weight)
        bias_number = 0
        m = len(y)
        for i in range(m):
            diff = y_hat[i]-y[i]
            for j in range(len(weight)):
                weight_number[j] += diff*x.iloc[i,j]
            bias_number += diff
        weight_number = np.array(weight_number)/m
        bias_number = bias_number/m
        
        temp_weight = weight - (alpha*weight_number)
        temp_bias = bias - alpha*bias_number
        
        weight = temp_weight
        bias = temp_bias

    return weight,bias,prediction_list,error_list


'''                               entring values of weight and bias                                                     '''
#initialize_weightandbias([-6.435226922178336, 1.6153259607066033, 4.863527658757547, 6.980279895857576, 6.446744004603078, -0.020494661629506944, 0.2196010512134854, -8.696576365823583, -0.010971070567052352, 0.38326127526701176, 0.44855660086911414, 7.046182919658131, 2.292357293872946],-4.9349192690255945)

'''                                               Traning the Model                                                        '''
if __name__ == "__main__":
    # provided initial weights & bias (use these instead of zeros)
    initial_weights = [
        -6.435226922178336, 1.6153259607066033, 4.863527658757547,
        6.980279895857576, 6.446744004603078, -0.020494661629506944,
        0.2196010512134854, -8.696576365823583, -0.010971070567052352,
        0.38326127526701176, 0.44855660086911414, 7.046182919658131,
        2.292357293872946
    ]
    initial_bias = -4.9349192690255945

    # sanity check: feature dimension must match
    if len(initial_weights) != x.shape[1]:
        raise ValueError(f"Initial weights length ({len(initial_weights)}) != number of features ({x.shape[1]})")

    # run gradient descent starting from these weights
    alpha = 0.01
    iterations = 1000
    trained_w, trained_b, preds, errors = gradientDescend(x, y, initial_weights, initial_bias, alpha, iterations)

    # save final trained parameters
    initialize_weightandbias(trained_w.tolist() if hasattr(trained_w, "tolist") else trained_w, float(trained_b))

    print("Training complete (started from provided initial weights).")
    print("Final bias:", trained_b)
    print("Final weights:", trained_w)
    print("Final error:", errors[-1])


'''                                         retreving weight and bias values                                               '''
def get_weightbias():
    weight_bias = pd.read_csv(r"weight_bias.csv")
    weight = weight_bias["weight"].tolist()
    bias = weight_bias["bias"][0]
    return weight,bias

def pppridct(features):
    features = pd.Series(features).astype(float)
    for i in range(len(max_values)):
        features[i] = features[i]/max_values[i]

    weight,bias = get_weightbias()
    output = predict_yi(features,weight,bias)

    if output>0.5:
        return 1
    return 0