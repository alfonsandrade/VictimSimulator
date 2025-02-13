import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging as log
import itertools
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import cm



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
y = dataset['grav']  

# Padronizando os dados (normalização)
# Para garantir que as variáveis estejam em uma escala similar, evitando que variáveis com maior amplitude de valores dominem o processo de aprendizado de máquina. Realiza a normalização dos dados com base na média e no desvio padrão, transformando os dados para que tenham uma média de 0 e desvio padrão de 1.
X = StandardScaler().fit_transform(X)

# Exibir as primeiras amostras
log.debug(f"O dataset possui (linhas, colunas): {dataset.shape}.")

# ***********************************************************************************************************************
# *****************************                        TRAINING MODEL                       *****************************
# ***********************************************************************************************************************
# NOTE: JUST FOR TRAINING: Separating dataset in training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
log.debug(f"Dados de treinamento X ({len(X_train)}):\n{X_train[:3]} ...")
log.debug(f"Dados de treinamento y ({len(y_train)}):\n{y_train[:3]} ...")
log.debug("-----------------------------------------------------------")
log.debug(f"Dados de teste X ({len(X_test)}):\n{X_test[:3]} ...")
log.debug(f"Dados de teste y ({len(y_test)}):\n{y_test[:3]} ...")

if Run == 'TRAINING':
    # Parameters' definition
    k_folds = 8
    max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    min_samples_leaf = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Generating all combinations of max_depth and min_samples_leaf
    param_combinations = list(itertools.product(max_depth, min_samples_leaf))

    # Auxiliar variables to store information
    best_model=[]   # store the best model of each parameterization
    model=[]        # store all models of all parameterizations
    train_scores=[] # store training scores
    vld_scores=[]   # store validation scores
    best_index=[]
    mse=[]
    best_validation_mse = float('inf')  # Initializing with a very high MSE to find the best one

    # Instantiating regressor until reach number of parameterizations
    for i, (depth, min_samples) in enumerate(param_combinations):
        regressor_tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples, random_state=42)

        # cross_validate function do fit and return, for each fold, the model learned, the train score and validation score
        training_results = cross_validate(
            regressor_tree,
            X_train, # Adjusting data format to be compatible with decision tree regressor function
            y_train,
            cv=k_folds,
            scoring='neg_mean_squared_error',  # In this case, is used mean_squared_error, it means that mse close to 0 are better
            return_train_score=True,  # Include training scores
            return_estimator=True    # Include trained models
        )

        # mean_squared_error of training and validation
        train_scores.append(training_results['train_score'])
        vld_scores.append(training_results['test_score'])

        # Calculate mean squared error (MSE) for training and validation
        train_mean_mse = -training_results['train_score'].mean()  # Invert neg_mean_squared_error to positive MSE
        valid_mean_mse = -training_results['test_score'].mean()   # Invert neg_mean_squared_error to positive MSE

        # Difference between training and validation mean squared error
        difference_mse = np.abs(train_scores[i] - vld_scores[i])

        # Store the index with the smallest difference_mse and the model with this index
        best_index.append(np.argmin(difference_mse))
        best_model.append(training_results['estimator'][best_index[i]])

        # If this model has the lowest validation error, store it
        if valid_mean_mse < best_validation_mse:
            best_validation_mse = valid_mean_mse
            best_off_all_model = training_results['estimator'][np.argmin(training_results['test_score'])]
            difference_score_best_validation = difference_mse[best_index[i]].any()
            best_max_depth = depth
            best_min_samples = min_samples

        # Store all parameterization models
        model.append(training_results['estimator'])

        # Printing all parameterization models and under it the index of the best model
        log.debug(f"Parametrization {i+1}: {model}")
        log.debug(f"Best Index: {best_index[i]}\n")

    # ***********************************************************************************************************************
    # *****************************                        Printing Results                       ***************************
    # ***********************************************************************************************************************
    log.debug("Train & Valid Scores (Neg mean_squared_error) per parametrization (Values close to 0 are better):")
    for i, (depth, min_samples) in enumerate(param_combinations):
        log.debug(f"Parameterization {i+1}\tMean\t\tScores per each k_fold")
        log.debug(f"Training:\t{train_scores[i].mean():>8.5f}\t{train_scores[i]}")
        log.debug(f"Validation:\t{vld_scores[i].mean():>8.5f}\t{vld_scores[i]}")
        log.debug(f"Diff MSE:\t{np.abs(train_scores[i].mean() - vld_scores[i].mean()):>8.5f}\t{abs(train_scores[i] - vld_scores[i])}")
        log.debug(f"Best index: {best_index[i]}\n")

    log.info("=========================================================")
    log.info(f"Number of parameterizations realized: {len(param_combinations)}\n")
    # Printing the best parameters and the best model VALIDATION MSE
    log.info("=========================================================")
    log.info(f"Best Validation Model: {best_off_all_model}")
    log.info(f"Best Validation Model MSE: {best_validation_mse:.5f}")
    log.info(f"Difference score for this Model: {difference_score_best_validation:.5f}\n")

    # initializing regressor with best parameters founded
    best_regressor_tree = DecisionTreeRegressor(
        max_depth=best_max_depth,
        min_samples_leaf=best_min_samples,
        random_state=42
    )

# ***********************************************************************************************************************
# *****************************                        TESTING MODEL                       ******************************
# ***********************************************************************************************************************
elif Run == 'TESTING':
    max_depth = 10          # Best max_depth founded when training
    min_samples_leaf = 3    # Best min_samples_leaf founded when training

    # initializing regressor with best parameters founded
    best_regressor_tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    # Setting just to follow pattern 
    X_test = X
    y_test = y

# ***********************************************************************************************************************
# ********************************                        RESULTS                       *********************************
# ***********************************************************************************************************************

# Training model with all dataset to fit
best_regressor_tree.fit(X_test, y_test)
# Realizing predictions with test data and calculating MSE
predictions = best_regressor_tree.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
log.info("=========== RESULTADO DO TESTE CEGO ===========")
log.info(f"Métricas - MSE: {mse}, MAE: {mae}, R2: {r2}\n")
# with open("regressor.pkl", "wb") as f:
#     pickle.dump(best_regressor_tree, f)

# # PLOT DAS MÉTRICAS
# # Criando os rótulos e valores
# metrics = ['MSE', 'MAE', 'R²']
# values = [mse, mae, r2]

# # Criando o gráfico de barras
# plt.figure(figsize=(8, 5))
# plt.bar(metrics, values, color=['blue', 'orange', 'green'])

# # Adicionando rótulos
# plt.ylabel("Metric Value")
# plt.title("Model Performance Metrics")
# plt.ylim(min(values) * 0.9, max(values) * 1.1)  # Ajusta os limites do eixo Y
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Adicionando valores em cima das barras
# for i, v in enumerate(values):
#     plt.text(i, v + (max(values) * 0.02), f"{v:.4f}", ha='center', fontsize=12)

# plt.show()


# ***********************************************************************************************************************
# *****************************                        Ploting Figures                       ****************************
# ***********************************************************************************************************************
# # Criar listas para armazenar as métricas
# train_mse = []
# vld_mse = []
# train_r2 = []
# vld_r2 = []
# # Calcular métricas para cada parametrização
# for i, (depth, min_samples) in enumerate(param_combinations):
#     train_mean_mse = -train_scores[i].mean()
#     valid_mean_mse = -vld_scores[i].mean()
#     train_mean_r2 = np.mean(1 - (train_scores[i] / np.var(y_train)))
#     valid_mean_r2 = np.mean(1 - (vld_scores[i] / np.var(y_test)))
    
#     train_mse.append(train_mean_mse)
#     vld_mse.append(valid_mean_mse)
#     train_r2.append(train_mean_r2)
#     vld_r2.append(valid_mean_r2)

# # Criar gráfico com múltiplas métricas
# plt.figure(figsize=(12, 6))
# param_labels = [f"D{d}-L{s}" for d, s in param_combinations]

# plt.plot(param_labels, train_mse, marker='o', label='Train MSE', linestyle='-')
# plt.plot(param_labels, vld_mse, marker='s', label='Validation MSE', linestyle='--')

# plt.xticks(rotation=45)
# plt.xlabel("Parametrização (Depth - Min Samples)")
# plt.ylabel("Metric Values")
# plt.title("Métricas - MSE")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# plt.plot(param_labels, train_r2, marker='o', label='Train R²', linestyle='-')
# plt.plot(param_labels, vld_r2, marker='s', label='Validation R²', linestyle='--')
# plt.xticks(rotation=45)
# plt.xlabel("Parametrização (Depth - Min Samples)")
# plt.ylabel("Metric Values")
# plt.title("Métricas - R²")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()
