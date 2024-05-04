import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = r"C:\Users\nesri\PycharmProjects\tp1\Book1.xlsx"
data=pd.read_excel(url)
print(data)
# Description des données
print(data.head())  # Afficher les premières lignes pour s'assurer que les données sont chargées correctement
print(data.describe())  # Résumé statistique des données
# Sélection des caractéristiques pour la classification
features = data.iloc[:, 1:]  # Sélection de toutes les colonnes sauf la première (nom du fromage)
# Standardisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# Classification ascendante hiérarchique (CAH)
linkage_matrix = linkage(features_scaled, method='ward', metric='euclidean')
# Affichage du dendrogramme
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('Dendrogramme CAH')
plt.xlabel('Fromages')
plt.ylabel('Distance')
plt.show()
# K-means
# Pistes pour le choix du nombre de clusters
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)
# Affichage du coude
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), inertia, marker='o')
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(features_scaled)
data['cluster_kmeans'] = kmeans.labels_
# Affichage des centres des clusters
print("Centres des clusters (K-means):")
print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features.columns))

# Sélection des caractéristiques pour la classification
# Standardisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
numeric_data = data.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap de la matrice de corrélation')
plt.show()
# Appliquer PCA
pca = PCA(n_components=2)  # Définir le nombre de composantes principales à 2
features_pca = pca.fit_transform(features_scaled)
# Classification avec K-means
kmeans = KMeans(n_clusters=4, random_state=42)  # Choisir le nombre de clusters
kmeans.fit(features_scaled)
cluster_labels = kmeans.labels_
# Affichage du nuage de points avec PCA et coloration par classe
plt.figure(figsize=(10, 6))
for i in range(len(set(cluster_labels))):  # Parcourez chaque classe unique
    plt.scatter(features_pca[cluster_labels == i, 0],  # Sélectionnez les données appartenant à la classe i
                features_pca[cluster_labels == i, 1],
                label=f'Cluster {i + 1}',  # Étiquette de la légende
                alpha=0.7)  # Opacité des points
plt.title('Nuage de points avec PCA (K-means clustering)')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.legend()
plt.show()

