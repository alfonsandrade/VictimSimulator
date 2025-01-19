import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging as log
import itertools

from sklearn.tree import DecisionTreeRegressor
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
Settings = 'PRESSURE'       # (You must use 'NOT_PRESSURE' or 'PRESSURE')

# Path to desired dataset
file_path = "datasets/data_800v/env_vital_signals.txt"

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
    best_difference_mse = float('inf')  # Initializing with a very high MSE to find the best one

    # Instantiating regressor until reach number of parameterizations
    for i, (depth, min_samples) in enumerate(param_combinations):
        regressor_tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples, random_state=42)

        # cross_validate function do fit and return, for each fold, the model learned, the train score and validation score
        training_results = cross_validate(
            regressor_tree,
            X_train.values.reshape(-1, X_train.shape[1]), # Adjusting data format to be compatible with decision tree regressor function
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
        
        # # If this model has the lowest difference error, store it
        # if difference_mse[best_index[i]].any() < best_difference_mse:
        #     best_difference_mse = difference_mse[best_index[i]].any()
        #     validation_score_best_mse = valid_mean_mse
        #     best_off_all_model_difference = training_results['estimator'][np.argmin(training_results['test_score'])]

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
    # # Printing the best parameters and the best model DIFFERENCE MSE
    # log.info("=========================================================")
    # log.info(f"Best Difference Model: {best_off_all_model_difference}")
    # log.info(f"Best Model Difference MSE: {best_difference_mse:.5f}")
    # log.info(f"Validation score for this Model: {validation_score_best_mse:.5f}\n")

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
    max_depth = 13          # Best max_depth founded when training
    min_samples_leaf = 2    # Best min_samples_leaf founded when training

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
log.info("=========== RESULTADO DO TESTE CEGO ===========")
log.info(f"Teste Cego - MSE: {mse}\n")

# ***********************************************************************************************************************
# *****************************                        Ploting Figures                       ****************************
# ***********************************************************************************************************************
# plt.figure(figsize=(10, 6))
# # For each parameterization, defines a new color using list comprehensions
# colors = [
#     [plt.cm.tab10(i * 2), plt.cm.tab10(i * 2 + 1)]
#     for i in range(num_params)
# ]

# for i in range(num_params):
#     # Ploting
#     plt.plot(range(1, len(train_scores[i]) + 1), train_scores[i], label=f"{i+1} Train Neg MSE", marker='o', color=colors[i][0])
#     plt.plot(range(1, len(vld_scores[i]) + 1), vld_scores[i], label=f"{i+1} Valid Neg MSE {i+1}", marker='o', color=colors[i][1])
#     plt.axhline(train_scores[i].mean(), color=colors[i][0], linestyle='--', label=f"{i+1} Train.mean {i+1}: {train_scores[i].mean():.2f}")
#     plt.axhline(vld_scores[i].mean(), color=colors[i][1], linestyle='--', label=f"{i+1} Valid.mean {i+1}: {vld_scores[i].mean():.2f}")

#     # Add labels and legend
#     plt.xlabel("Fold")
#     plt.ylabel("Neg MSE")
#     plt.title("Training and Validation Scores (Bias and Variance)")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.xticks(np.arange(1, k_folds + 1, 1))
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

# ***********************************************************************************************************************
# **********                        Ploting decisions tree for each parameterization                       **************
# ***********************************************************************************************************************
# fig, axes = plt.subplots(1, num_params, figsize=(16, 6))  # 1 row, num_params columns

# for i in range(num_params):
#     tree.plot_tree(best_model[i],
#                    feature_names = X.columns.tolist(),
#                    filled=True,
#                    rounded=True,
#                    class_names=True,
#                    fontsize=8,
#                    ax=axes[i])  
#     axes[i].set_title(f"Parameterization {i+1}")  # Identify each parameterization

# plt.tight_layout()  # Adjust spacing between subplots
# plt.show()