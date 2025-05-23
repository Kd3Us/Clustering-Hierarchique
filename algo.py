import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les visualisations
plt.style.use('default')
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f']
custom_cmap = ListedColormap(colors)

print("Démonstration du Clustering Hiérarchique sur des confessions Reddit")

def clean_text(text):
    if text is None or text == "[removed]" or text == "[deleted]":
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenisation simple par espace
    tokens = text.split()
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
                  "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
                  "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
                  "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
                  "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", 
                  "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", 
                  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
                  "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", 
                  "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
                  "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", 
                  "should", "now"}
    # Filtrage des stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
print("\nÉtape 1: Chargement et préparation des données")

print("Chargement du dataset...")
dataset = load_dataset("SocialGrep/one-million-reddit-confessions")
df = pd.DataFrame(dataset['train'])
print(f"Dataset chargé! Nombre total de confessions: {len(df)}")

# Filtrage des confessions non vides
df = df[~df['selftext'].isin(['[removed]', '[deleted]', None, ''])]
print(f"Après filtrage: {len(df)} confessions valides")

sample_size = 50
np.random.seed(42)  # Pour la reproductibilité
sample_df = df.sample(sample_size)
print(f"Échantillon pour la démonstration: {sample_size} confessions")

# Nettoyage des textes
print("Nettoyage des textes...")
sample_df['clean_text'] = sample_df['selftext'].apply(clean_text)

sample_df['short_title'] = sample_df['title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)

sample_df = sample_df.reset_index(drop=True)

# 2. VECTORISATION DES TEXTES
print("\nÉtape 2: Vectorisation des textes avec TF-IDF")
print("----------------------------------------------")

vectorizer = TfidfVectorizer(max_features=100, min_df=2, stop_words='english')
X = vectorizer.fit_transform(sample_df['clean_text'])
X_dense = X.toarray()
print(f"Textes vectorisés: {X.shape[0]} documents, {X.shape[1]} caractéristiques")

# Calcul de la matrice de similarité/distance
similarity = cosine_similarity(X)
distance = 1 - similarity
print("Matrice de distance calculée!")

# 3. VISUALISATION DU DENDROGRAMME
print("\nÉtape 3: Création du dendrogramme")
print("----------------------------------------------")

# Calcul du linkage pour le dendrogramme
Z = linkage(distance, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(
    Z,
    labels=sample_df['short_title'].values,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=0.7*max(Z[:,2])  # Coloration des clusters
)
plt.title('Dendrogramme du Clustering Hiérarchique des Confessions Reddit', fontsize=16)
plt.xlabel('Confessions', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
print("Dendrogramme créé et enregistré comme 'dendrogram.png'")

# 4. COMPARAISON DES MÉTHODES DE LIAISON
print("\nÉtape 4: Comparaison des différentes méthodes de liaison")
print("----------------------------------------------")

methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, method in enumerate(methods):
    Z = linkage(distance, method=method)
    dendrogram(Z, ax=axes[i], labels=None, color_threshold=0.7*max(Z[:,2]))
    axes[i].set_title(f'Méthode: {method}', fontsize=14)
    if i >= 2:  # Ajouter les labels x seulement pour les graphiques du bas
        axes[i].set_xlabel('Index des confessions', fontsize=10)
    axes[i].set_ylabel('Distance', fontsize=10)

plt.tight_layout()
plt.savefig('linkage_methods.png', dpi=300, bbox_inches='tight')
print("Comparaison des méthodes de liaison créée et enregistrée comme 'linkage_methods.png'")

# 5. DÉTERMINATION DU NOMBRE OPTIMAL DE CLUSTERS
print("\nÉtape 5: Détermination du nombre optimal de clusters")
print("----------------------------------------------")

max_clusters = 10
distances = np.arange(1, max_clusters + 1)
num_clusters = []

for d in distances:
    clusters = fcluster(Z, t=d, criterion='distance')
    num_clusters.append(len(np.unique(clusters)))

# Visualisation du nombre de clusters en fonction de la distance de coupure
plt.figure(figsize=(10, 6))
plt.plot(distances, num_clusters, marker='o', linestyle='-', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Distance de coupure', fontsize=12)
plt.ylabel('Nombre de clusters', fontsize=12)
plt.title('Nombre de clusters en fonction de la distance de coupure', fontsize=16)
plt.xticks(distances)
plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
print("Graphique du nombre optimal de clusters créé et enregistré comme 'optimal_clusters.png'")

# 6. APPLICATION DU CLUSTERING AVEC UN NOMBRE SPÉCIFIQUE DE CLUSTERS
print("\nÉtape 6: Application du clustering hiérarchique")
print("----------------------------------------------")

# Définir le nombre de clusters basé sur l'observation du dendrogramme
n_clusters = 5
print(f"Nombre de clusters choisi: {n_clusters}")

# Application du clustering hiérarchique
model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
clusters = model.fit_predict(distance)

sample_df['cluster'] = clusters

cluster_counts = sample_df['cluster'].value_counts().sort_index()
print("Nombre de confessions par cluster:")
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} confessions")

# 7. VISUALISATION DES CLUSTERS AVEC T-SNE
print("\nÉtape 7: Visualisation des clusters avec t-SNE")
print("----------------------------------------------")

tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, sample_size-1))
X_tsne = tsne.fit_transform(X_dense)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=clusters, 
                     cmap=custom_cmap, 
                     s=100, 
                     alpha=0.8,
                     edgecolors='w')

# Ajouter les étiquettes des clusters
for cluster_id in range(n_clusters):
    cluster_points = X_tsne[clusters == cluster_id]
    if len(cluster_points) > 0:
        centroid = cluster_points.mean(axis=0)
        plt.annotate(f'Cluster {cluster_id}', 
                    xy=(centroid[0], centroid[1]),
                    xytext=(centroid[0], centroid[1]),
                    fontsize=12, 
                    fontweight='bold',
                    color='black',
                    ha='center')

plt.title('Visualisation des clusters de confessions Reddit avec t-SNE', fontsize=16)
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('tsne_clusters.png', dpi=300, bbox_inches='tight')
print("Visualisation t-SNE créée et enregistrée comme 'tsne_clusters.png'")

# 8. ANALYSE DES CLUSTERS
print("\nÉtape 8: Analyse des clusters et des mots caractéristiques")
print("----------------------------------------------")

# Extraction des mots les plus caractéristiques de chaque cluster
feature_names = vectorizer.get_feature_names_out()

with open('cluster_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("ANALYSE DES CLUSTERS DE CONFESSIONS REDDIT\n")
    f.write("========================================\n\n")
    
    for cluster_id in range(n_clusters):
        f.write(f"\nCLUSTER {cluster_id}:\n")
        f.write("-" * 30 + "\n")
        
        cluster_mask = sample_df['cluster'] == cluster_id
        
        f.write("\nMots caractéristiques:\n")
        
        if cluster_mask.sum() > 0:
            cluster_indices = np.arange(len(sample_df))[cluster_mask]
            
            cluster_vectors = X[cluster_indices]
            
            cluster_tfidf_mean = cluster_vectors.mean(axis=0)
            cluster_tfidf = np.asarray(cluster_tfidf_mean).flatten()
            
            top_indices = cluster_tfidf.argsort()[-10:][::-1]
            
            for idx in top_indices:
                if cluster_tfidf[idx] > 0:
                    f.write(f"  {feature_names[idx]}: {cluster_tfidf[idx]:.4f}\n")
        
        f.write("\nTitres des confessions:\n")
        cluster_confessions = sample_df[cluster_mask]
        for _, row in cluster_confessions.iterrows():
            f.write(f"  - {row['title']}\n")
        
        f.write("\nExemple de confession:\n")
        if len(cluster_confessions) > 0:
            example = cluster_confessions.iloc[cluster_confessions['selftext'].str.len().argmin()]
            text = example['selftext']
            if len(text) > 300:
                text = text[:300] + "..."
            f.write(f"{text}\n")

print("Analyse des clusters terminée et enregistrée dans 'cluster_analysis.txt'")

print("\nAnalyse terminée avec succès!")
print("=================================================================")
print("Fichiers générés:")
print("  - dendrogram.png: Dendrogramme initial")
print("  - linkage_methods.png: Comparaison des méthodes de liaison")
print("  - optimal_clusters.png: Analyse du nombre optimal de clusters")
print("  - tsne_clusters.png: Visualisation des clusters avec t-SNE")
print("  - cluster_analysis.txt: Analyse détaillée des clusters")
print("\nVous pouvez maintenant utiliser ces visualisations pour votre présentation!")