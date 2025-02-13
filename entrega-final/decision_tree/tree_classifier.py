import pandas as pd
import matplotlib.pyplot as plt
import logging as log
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# ***********************************************************************************************************************
# ***************************                        ADJUSTING DATASET                       ****************************
# ***********************************************************************************************************************
# Setting Log
log.basicConfig(level = log.INFO, format='%(message)s')

# Setting Current RUN goal    
Run = 'TESTING'                # (You must use 'TRAINING' or 'TESTING')
Settings = 'NOT_PRESSURE'       # (You must use 'NOT_PRESSURE' or 'PRESSURE')

# Path to desired dataset
file_path = "datasets/data_4000v/env_vital_signals.txt"

# Loading desired dataset
dataset = pd.read_csv(file_path, sep=',', header=None)

# Naming Columns
dataset.columns = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'fResp', 'grav', 'label']

# Separating in columns to decision tree use (X) and Column to predict (Y). 
# NOTE: Its excluding 'grav' column
if Settings == 'PRESSURE':
    X = dataset[['pSist', 'pDiast', 'qPA', 'pulso', 'fResp']]
elif Settings == 'NOT_PRESSURE':
    X = dataset[['qPA', 'pulso', 'fResp']]
y = dataset['label']  

# Padronizando os dados (normalização)
# Para garantir que as variáveis estejam em uma escala similar, evitando que variáveis com maior amplitude de valores dominem o processo de aprendizado de máquina. Realiza a normalização dos dados com base na média e no desvio padrão, transformando os dados para que tenham uma média de 0 e desvio padrão de 1.
X = StandardScaler().fit_transform(X)

# Exibir as primeiras amostras
log.debug(f"O dataset possui (linhas, colunas): {dataset.shape}.")

# ***********************************************************************************************************************
# *****************************                        TRAINING MODEL                       *****************************
# ***********************************************************************************************************************
if Run == 'TRAINING':
    # NOTE: JUST FOR TRAINING: Separating dataset in training and testing 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    log.debug(f"Dados de treinamento X ({len(X_train)}):\n{X_train[:3]} ...")
    log.debug(f"Dados de treinamento y ({len(y_train)}):\n{y_train[:3]} ...")
    log.debug("-----------------------------------------------------------")
    log.debug(f"Dados de teste X ({len(X_test)}):\n{X_test[:3]} ...")
    log.debug(f"Dados de teste y ({len(y_test)}):\n{y_test[:3]} ...")

    # Parameters' definition
    parameters = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    # instantiate model with random_state = 42 to be deterministic. Check https://scikit-learn.org/stable/glossary.html#term-random_state
    model = DecisionTreeClassifier(random_state=42)

    # grid search using cross-validation
    # cv = 3 is the number of folds
    # 'f1' = 2 * (precision * recall) / (precision + recall)
    # scoring the metric = 'f1' for chosing the best model. Check https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    scoring = GridSearchCV(model, parameters, cv=3, scoring='f1_weighted', verbose=0) # verbose define log level [0, 4]
    scoring.fit(X_train, y_train)



#===========================================================================================================================
        # Salvando os resultados de cada combinação testada
    results_df = pd.DataFrame(scoring.cv_results_)
    results_df.to_csv("decision_tree_results.csv", index=False)
    
    log.info("Resultados da busca por hiperparâmetros salvos em 'decision_tree_results.csv'")
    # Plotando os resultados para MAX_DEPTH
    plt.figure(figsize=(10, 6))
    for criterion in ['entropy', 'gini']:
        subset = results_df[results_df['param_criterion'] == criterion]
        plt.plot(subset['param_max_depth'], subset['mean_test_score'], marker='o', label=f'Criterion: {criterion}')
    
    plt.xlabel('Max Depth')
    plt.ylabel('F1-Weighted Score')
    plt.title('Desempenho do Decision Tree para Diferentes Profundidades')
    plt.legend()
    plt.grid()
    plt.show()

        # Plotando os resultados para min_samples_leaf
    plt.figure(figsize=(10, 6))
    for criterion in ['entropy', 'gini']:
        subset = results_df[results_df['param_criterion'] == criterion]
        plt.plot(subset['param_min_samples_leaf'], subset['mean_test_score'], marker='s', label=f'Criterion: {criterion}')
    
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('F1-Weighted Score')
    plt.title('Desempenho do Decision Tree para Diferentes Min Samples Leaf')
    plt.legend()
    plt.grid()
    plt.show()
#===========================================================================================================================



    # The best tree according to the scoring realized
    best_scoring = scoring.best_estimator_
    log.info("\n===== Melhor classificador =====")
    log.info(scoring.best_estimator_)
    log.info("\n===== Parametros do classificador =====")
    log.info(scoring.best_estimator_.get_params())

    # Getting acurracy results using TRAINING data
    y_pred_train = best_scoring.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train) * 100
    log.info(f"Acuracia com dados de treino: {acc_train:.2f}%\n")

    # Getting acurracy results using TESTING data (not used before)
    y_pred_test = best_scoring.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test) * 100
    log.info(f"Acuracia com dados de teste: {acc_test:.2f}%\n")

# ***********************************************************************************************************************
# *****************************                        TESTING MODEL                       ******************************
# ***********************************************************************************************************************
elif Run == 'TESTING':

    # Seting classifier with best founded parameters when running in TRAINING mode
    best_scoring = DecisionTreeClassifier(
        criterion = 'entropy',
        max_depth = 11,
        min_samples_leaf = 2,
        random_state = 42
    )

    # Setting just to follow pattern 
    X_test = X
    y_test = y

    # Getting acurracy results using all dataset data
    best_scoring.fit(X_test, y_test)
    y_pred_test = best_scoring.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test) * 100
    log.info("=========== RESULTADO DO TESTE CEGO ===========")
    log.info(f"Acuracia com dados de teste cego: {acc_test:.2f}%\n")
    # with open("classifier.pkl", "wb") as f:
    #     pickle.dump(best_scoring, f)
    

# If you want to see DECISION TREE used
# from sklearn import tree
# fig = plt.figure(figsize=(8, 6))
# if Settings == 'PRESSURE':
#     tree.plot_tree(best_scoring, feature_names=['pSist', 'pDiast', 'qPA', 'pulso', 'fResp'], filled=True, rounded=True, class_names=True, fontsize=8)
# elif Settings == 'NOT_PRESSURE':
#     tree.plot_tree(best_scoring, feature_names=['qPA', 'pulso', 'fResp'], filled=True, rounded=True, class_names=True, fontsize=8)
# plt.show()

# If you want to see CONFUSION MATRIX
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
plt.show()
print(classification_report(y_test, y_pred_test))