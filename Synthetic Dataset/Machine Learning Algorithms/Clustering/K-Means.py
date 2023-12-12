import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time 

# Enregistrez le temps de début
start_time = time.time()

# Chargement des données depuis Kaggle (assurez-vous que le fichier est dans le même répertoire que le script)
df = pd.read_csv("Synthetic Dataset/SyntheticDataset.csv")

df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Séparation des caractéristiques (X)
# Prépare mes données pour l'entraînement d'un modèle
X = df

# Mise à l'échelle des données permet de d'assurer que chaque caractéristiques contribuent de manière équitable au modèle.
# Normaliser des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Création d'un modèle K-Means avec 3 clusters (à adapter selon le besoin)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init

# Entraînement du modèle
kmeans.fit(X_scaled)

# Prédictions des clusters sur les données
labels = kmeans.labels_

# Ajout des labels des clusters à notre DataFrame
df['cluster'] = labels

# Enregistrez le temps de fin
end_time = time.time()
# Calculez le temps d'exécution total
execution_time = end_time - start_time

print(f'\nTemps d\'exécution total: {execution_time} secondes')

# Visualisation des clusters dans l'espace 2D (à adapter selon le besoin)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title('Clusters formés par K-Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

