import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.utils import resample
import time

# Enregistrez le temps de début
start_time = time.time()

# Charger les données depuis le fichier CSV
df = pd.read_csv("Large Dataset/CleanLargeDataset.csv")

df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Sélectionner les features pour le clustering (exclure la variable cible 'isFraud')
X = df.drop('isFraud', axis=1)

# Sélectionner un sous-ensemble des données pour réduire la taille
sample_size = 1000
data_sample = resample(X, n_samples=sample_size, random_state=42)

# Création du modèle de clustering hiérarchique
linkage_matrix = linkage(data_sample, method='ward')

# Enregistrez le temps de fin
end_time = time.time()
# Calculez le temps d'exécution total
execution_time = end_time - start_time

print(f'\nTemps d\'exécution total: {execution_time} secondes')

# Affichage du dendrogramme
plt.figure(figsize=(15, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Dendrogramme Hierarchical Clustering')
plt.xlabel('Indices des échantillons')
plt.ylabel('Distance euclidienne')
plt.show()
