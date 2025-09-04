# Heart Disease Prediction 🫀  

This project implements a **binary classification model** to predict the presence of heart disease using medical data.  

It includes:  
- **Model training** using Logistic Regression implemented from scratch with NumPy and Pandas.  
- **Flask API** to serve predictions.  
- **MongoDB integration** for storing datasets and trained model parameters.  

---

## Project Structure  
```
HEART-DISEASE-PREDICTION/
├─- Model-training-github/ # training & model code (typo: consider renaming)
|       ├─- Binary_Classification.py # Train logistic regression from scratch
|       ├─- Heart_Disease_Prediction.csv # original dataset (from Kaggle)
|       ├-─ training_dataSet.csv # Training data
|       ├-─ testing_dataSet.csv # Testing data
|       └-- rained weights and bias
|
|─- Logistical_Regression.py # Prediction helper functions
├─- API_File.py # Flask API for predictions
├─- Store_to_mongoDB.py # Upload data & model to MongoDB
|-- requirements.txt # Dependencies
└─- README.md # Documentation
```

---

## ⚙Setup Instructions  

1. **Clone the repository**
   ```bash
   git clone https://github.com/engg-TarsemSingh/HEART-DISEASE-PREDICTION.git
   cd HEART-DISEASE-PREDICTION
   ```
   
2. **Create virtual environment**
   ```bash
    python3 -m venv venv
    source venv/bin/activate   # for Linux/Mac
    venv\Scripts\activate      # for Windows
   ```
3. **Install dependencies**
  ```bash
    pip install -r requirements.txt
  ```
4. **Run training**
   ```bash
   python3 model_training/Binary_Classification.py

  This will:
  Preprocess dataset
  Train logistic regression with gradient descent
  Save trained weights & bias into weight_bias.csv

5. **Store to MongoDB**
   ```bash
   python3 Store_to_mongoDB.py

  Make sure to set your MongoDB connection string in Store_to_mongoDB.py:
  ```python
    URL = "your_mongodb_connection_url"
```

6. **Run API**
   ```bash
    python3 API_File.py

