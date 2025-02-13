import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def decision_tree_regressor(Settings, dataset):
    # Adjusting to pandas dataset and naming Columns
    dataset = pd.DataFrame(dataset, columns=['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'fResp'])

    # Separating in columns to decision tree use (X) and Column to predict (Y). 
    X = dataset[['pSist', 'pDiast', 'qPA', 'pulso', 'fResp']] if Settings == 'PRESSURE' else dataset[['qPA', 'pulso', 'fResp']]

    # Normalizing the data
    X = StandardScaler().fit_transform(X)

    # Loading classifier with best founded parameters 
    with open("regressor.pkl", "rb") as f:
        best_regressor_tree = pickle.load(f)

    # Realizing predictions with data
    predictions = best_regressor_tree.predict(X)
    # print(predictions)
    return predictions
