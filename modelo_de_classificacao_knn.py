import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Verificar se o arquivo existe
if not os.path.exists('EMGDataset.csv'):
    print("ERRO Arquivo EMGDataset.csv não encontrado!")
    exit()

# Carregamento dos dados
data = pd.read_csv('EMGDataset.csv')
print(f"Dataset carregado com sucesso! Shape: {data.shape}")

# X ∈ R^(N×p) para MQO (N=50000, p=2)
# As duas primeiras colunas são os sensores

X = data.iloc[:, :2].values  # Corrugador do Supercílio e Zigomático Maior
y = data.iloc[:, 2].values   # Labels das classes

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Mapeamento das classes
class_names = {1: 'Neutro', 2: 'Sorriso', 3: 'Sobrancelhas levantadas', 4: 'Surpreso', 5: 'Rabugento'}
print("\nClasses no dataset:")
for class_id, class_name in class_names.items():
    count = np.sum(y == class_id)
    print(f"{class_id} - {class_name}: {count} amostras")

# 1. Visualização inicial dos dados
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, (class_id, class_name) in enumerate(class_names.items()):
    mask = y == class_id
    plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'{class_name}', alpha=0.6, s=1)

plt.xlabel('Corrugador do Supercílio (Sensor 1)')
plt.ylabel('Zigomático Maior (Sensor 2)')
plt.title('Visualização dos Dados EMG por Classe')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Análise de separabilidade linear
print("\nAnálise de separabilidade:")
print("Baseado na visualização, podemos observar se as classes são linearmente separáveis ou não.")

# Validaçao cruzada k-fold para encontrar o melhor K
k_values = [1, 7, 11, 17, 23, 39, 101]
cv_scores = []
cv_std = []

print("\nTeste de diferentes valores de K com k-fold cross-validation:")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
    cv_scores.append(scores.mean())
    cv_std.append(scores.std())
    print(f"K={k}: Acurácia média = {scores.mean():.4f} (±{scores.std():.4f})")

# Encontrar o melhor K
best_k_idx = np.argmax(cv_scores)
best_k = k_values[best_k_idx]
print(f"\nMelhor K encontrado: {best_k} com acurácia de {cv_scores[best_k_idx]:.4f}")

# Visualizar resultados do k-fold
plt.figure(figsize=(10, 6))
plt.errorbar(k_values, cv_scores, yerr=cv_std, marker='o', capsize=5)
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.title('K-NN: Acurácia vs Valor de K (k-fold cross-validation)')
plt.grid(True)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Melhor K = {best_k}')
plt.legend()
plt.show()

# Validação por amostragem aleatória com o melhor K
num_rodadas = 500
teste_percent = 0.2
treino_percent = 1 - teste_percent

accuracy_list = []
confusion_matrices = []

print(f"\nExecutando {num_rodadas} rodadas de validação com K={best_k}...")

for i in range(num_rodadas):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=teste_percent, random_state=i)

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

# Cálculo das métricas estatísticas
accuracy_array = np.array(accuracy_list)
media_accuracy = accuracy_array.mean()
std_accuracy = accuracy_array.std()
max_accuracy = accuracy_array.max()
min_accuracy = accuracy_array.min()

print(f"\nResultados após {num_rodadas} rodadas:")
print(f"Acurácia - Média: {media_accuracy:.4f}, Desvio Padrão: {std_accuracy:.4f}")
print(f"Acurácia - Máximo: {max_accuracy:.4f}, Mínimo: {min_accuracy:.4f}")

# Encontrar as matrizes de confusão com maior e menor acurácia
idx_max_acc = np.argmax(accuracy_list)
idx_min_acc = np.argmin(accuracy_list)

cm_max = confusion_matrices[idx_max_acc]
cm_min = confusion_matrices[idx_min_acc]

print(f"\nMatriz de Confusão - Maior Acurácia ({accuracy_list[idx_max_acc]:.4f}):")
print(cm_max)

print(f"\nMatriz de Confusão - Menor Acurácia ({accuracy_list[idx_min_acc]:.4f}):")
print(cm_min)

# Visualização das matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Matrix com maior acuracia
sns.heatmap(cm_max, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(class_names.values()),
            yticklabels=list(class_names.values()),
            ax=axes[0])
axes[0].set_title(f'Matriz de Confusão - Maior Acurácia ({accuracy_list[idx_max_acc]:.4f})')
axes[0].set_xlabel('Predito')
axes[0].set_ylabel('Real')

# Matrix com menor acuacia
sns.heatmap(cm_min, annot=True, fmt='d', cmap='Reds',xticklabels=list(class_names.values()), yticklabels=list(class_names.values()),ax=axes[1])
axes[1].set_title(f'Matriz de Confusão - Menor Acurácia ({accuracy_list[idx_min_acc]:.4f})')
axes[1].set_xlabel('Predito')
axes[1].set_ylabel('Real')
plt.tight_layout()
plt.show()

# Visualização da distribuiçao das acuracias
plt.figure(figsize=(10, 6))
plt.hist(accuracy_list, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.title('Distribuição das Acurácias')
plt.axvline(media_accuracy, color='red', linestyle='--', label=f'Média: {media_accuracy:.4f}')
plt.legend()
plt.grid(True)
plt.show()

# Salvar resultados
resultados_dir = 'resultados_classificacao'
if not os.path.exists(resultados_dir):
    os.makedirs(resultados_dir)

# Salvar metricas
resultados_path = os.path.join(resultados_dir, 'resultados_knn.txt')
with open(resultados_path, 'w') as f:
    f.write(f"Resultados K-NN com EMG Dataset\n")
    f.write(f"================================\n\n")
    f.write(f"Melhor K encontrado: {best_k}\n")
    f.write(f"Número de rodadas: {num_rodadas}\n")
    f.write(f"Particionamento: {treino_percent*100:.0f}% treino, {teste_percent*100:.0f}% teste\n\n")
    f.write(f"Métricas de Acurácia:\n")
    f.write(f"Média: {media_accuracy:.4f}\n")
    f.write(f"Desvio Padrão: {std_accuracy:.4f}\n")
    f.write(f"Maior Valor: {max_accuracy:.4f}\n")
    f.write(f"Menor Valor: {min_accuracy:.4f}\n\n")
    f.write(f"Matriz de Confusão - Maior Acurácia ({accuracy_list[idx_max_acc]:.4f}):\n")
    f.write(str(cm_max))
    f.write(f"\n\nMatriz de Confusão - Menor Acurácia ({accuracy_list[idx_min_acc]:.4f}):\n")
    f.write(str(cm_min))

print(f"\nResultados salvos em: {resultados_path}")