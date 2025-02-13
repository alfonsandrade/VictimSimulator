# Rede Neural Classificador
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Carregando os dados
data = pd.read_csv('../../datasets/data_4000v/env_vital_signals.txt', header=None)
data.columns = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'fResp', 'grav', 'label']

# Separando as características (qPA, pulso, FRPM) e a variável alvo (Label)
X = data[['qPA', 'pulso', 'fResp']]
# print(f'X:\n{X}\n')

y = data['label']
# print(f'Y:\n{y}\n')

# Padronizando os dados (normalização)
# Para garantir que as variáveis estejam em uma escala similar, evitando que variáveis com maior amplitude de valores dominem o processo de aprendizado de máquina. Realiza a normalização dos dados com base na média e no desvio padrão, transformando os dados para que tenham uma média de 0 e desvio padrão de 1.
X = StandardScaler().fit_transform(X)
# print(f'X Normalizado:\n{X}\n')

# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# Definindo as parametrizações
param_grid = {
	'hidden_layer_sizes': [(20,10,10,10,10), (10,10,20,10,10), (10,10,10,10,20)]
}

# Criando a rede neural
mlp = MLPClassifier(
		max_iter=2000,
		tol=0.000001,
		solver='adam',
		activation='relu'
		)

# Realizar a validação cruzada
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', verbose=True)
grid_search.fit(X_train, y_train)

# Melhor modelo
best_model = grid_search.best_estimator_
print(f"\nMelhores parâmetros: {grid_search.best_params_}")

# Avaliando o modelo
y_pred = best_model.predict(X_test)
# print(f'Y predito: {y_pred}')
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred))

# Gerando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Salvando o modelo treinado
# joblib.dump(best_model, 'MLP-classificador.pkl')
