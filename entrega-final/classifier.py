import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def decision_tree_classifier(Settings, dataset):
    # Adjusting to pandas dataset and naming Columns
    dataset = pd.DataFrame(dataset, columns=['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'fResp'])

    # Separating in columns to decision tree use
    X = dataset[['pSist', 'pDiast', 'qPA', 'pulso', 'fResp']] if Settings == 'PRESSURE' else dataset[['qPA', 'pulso', 'fResp']]

    # Normalizing the data
    X = StandardScaler().fit_transform(X)

    # Loading classifier with best founded parameters 
    with open("classifier.pkl", "rb") as f:
        best_classifier_tree = pickle.load(f)

    # Realizing predictions with data
    prediction = best_classifier_tree.predict(X)
    # print(prediction)
    return prediction