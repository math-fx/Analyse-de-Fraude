import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# Enregistrez le temps de début
start_time = time.time()

# Charger les données depuis le fichier CSV
df = pd.read_csv("Large Dataset/CleanLargeDataset.csv")

df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Séparer les données en features (X) et la variable cible (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Diviser le dataset en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser le modèle de régression Ridge
model = Ridge(alpha=1.0, random_state=42)  # Vous pouvez ajuster le paramètre alpha selon vos besoins

# Entraîner le modèle
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Convertir les prédictions continues en classes binaires (0 ou 1) avec un seuil de 0.5
threshold = 0.5
y_pred_binary = [1 if x >= threshold else 0 for x in y_pred]

# Évaluer les performances du modèle pour la classification
matrix = confusion_matrix(y_test, y_pred_binary)
accuracy = accuracy_score(y_test, y_pred_binary)
report = classification_report(y_test, y_pred_binary)

# Enregistrez le temps de fin
end_time = time.time()
# Calculez le temps d'exécution total
execution_time = end_time - start_time

# Affichage des résultats
print("Matrice d'erreur:\n", matrix)
print("Rapport de classification:\n", report)
print(f"Précision: {accuracy:.4f}")
print(f'\nTemps d\'exécution total: {execution_time} secondes')
