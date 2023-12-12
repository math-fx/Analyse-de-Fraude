import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Enregistrez le temps de début
start_time = time.time()

# Charger les données depuis le fichier CSV
df = pd.read_csv("Synthetic Dataset/SyntheticDataset.csv")

df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Séparer les données en features (X) et la variable cible (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle GMM
gmm = GaussianMixture(n_components=2, random_state=42)  # 2 composants pour les classes frauduleuses et non frauduleuses
gmm.fit(X_train)

# Prédire les labels sur l'ensemble de test
y_pred = gmm.predict(X_test)

# Évaluer les performances du modèle
matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Enregistrez le temps de fin
end_time = time.time()
# Calculez le temps d'exécution total
execution_time = end_time - start_time

# Affichage des résultats
print("Matrice d'erreur:\n", matrix)
print("Rapport de classification:\n", report)
print(f"Précision: {accuracy:.4f}")
print(f'\nTemps d\'exécution total: {execution_time} secondes')
