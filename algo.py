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

print("Démonstration du Clustering Hiérarchique sur des confessions Reddit")

def clean_text(text):
    if text is None or text == "[removed]" or text == "[deleted]":
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
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
np.random.seed(42)
sample_df = df.sample(sample_size)
print(f"Échantillon pour la démonstration: {sample_size} confessions")

# Nettoyage des textes
print("Nettoyage des textes...")
sample_df['clean_text'] = sample_df['selftext'].apply(clean_text)
sample_df['short_title'] = sample_df['title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
sample_df = sample_df.reset_index(drop=True)

# 2. VECTORISATION DES TEXTES
print("\nÉtape 2: Vectorisation des textes avec TF-IDF")
vectorizer = TfidfVectorizer(max_features=100, min_df=2, stop_words='english')
X = vectorizer.fit_transform(sample_df['clean_text'])
X_dense = X.toarray()
print(f"Textes vectorisés: {X.shape[0]} documents, {X.shape[1]} caractéristiques")

# Calcul de la matrice de similarité/distance
similarity = cosine_similarity(X)
distance = 1 - similarity
print("Matrice de distance calculée!")

# 3. CLUSTERING AUTOMATIQUE (non supervisé)
print("\nÉtape 3: Clustering hiérarchique automatique")
Z = linkage(distance, method='ward')

# Détermination automatique du nombre de clusters optimal
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import squareform

# Test différents nombres de clusters
silhouette_scores = []
max_clusters = min(10, sample_size//2)

for n_clusters in range(2, max_clusters):
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Calculer un score de qualité simple basé sur la cohésion intra-cluster
    cluster_distances = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 1:
            cluster_dist = distance[np.ix_(cluster_indices, cluster_indices)]
            cluster_distances.append(np.mean(cluster_dist))
    
    if cluster_distances:
        silhouette_scores.append(np.mean(cluster_distances))
    else:
        silhouette_scores.append(float('inf'))

# Choisir le nombre optimal de clusters
optimal_clusters = np.argmin(silhouette_scores) + 2
print(f"Nombre optimal de clusters déterminé automatiquement: {optimal_clusters}")

# Obtenir les clusters finaux
final_clusters = fcluster(Z, optimal_clusters, criterion='maxclust')
sample_df['cluster'] = final_clusters

# 4. CRÉATION DU DENDROGRAMME AVEC HEATMAP DES MOTS-CLÉS
print("\nÉtape 4: Création du dendrogramme complet avec heatmap des mots")

# Extraire les mots caractéristiques pour chaque cluster
feature_names = vectorizer.get_feature_names_out()
top_words_per_cluster = {}
n_top_words = 10

# Créer une matrice des mots les plus importants par cluster
cluster_word_matrix = np.zeros((optimal_clusters, n_top_words))
word_labels = []

for cluster_id in range(1, optimal_clusters + 1):
    cluster_mask = sample_df['cluster'] == cluster_id
    
    if cluster_mask.sum() > 0:
        cluster_indices = np.where(cluster_mask)[0]
        cluster_vectors = X[cluster_indices]
        cluster_tfidf_mean = cluster_vectors.mean(axis=0)
        cluster_tfidf = np.asarray(cluster_tfidf_mean).flatten()
        
        # Obtenir les indices des mots les plus importants
        top_indices = cluster_tfidf.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        top_scores = cluster_tfidf[top_indices]
        
        # Stocker pour la heatmap
        cluster_word_matrix[cluster_id-1, :] = top_scores
        
        # Stocker les mots pour les labels (seulement pour le premier cluster pour éviter la répétition)
        if cluster_id == 1:
            word_labels = top_words
        
        top_words_per_cluster[cluster_id] = list(zip(top_words, top_scores))

# Création de la figure avec deux sous-graphiques
fig = plt.figure(figsize=(20, 12))

# Créer une grille pour organiser les sous-graphiques
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

# Sous-graphique 1: Dendrogramme
ax1 = fig.add_subplot(gs[0])
dend = dendrogram(
    Z,
    labels=[f"Cluster_{cluster}" for cluster in final_clusters],
    leaf_rotation=90,
    leaf_font_size=10,
    ax=ax1,
    color_threshold=0.7*max(Z[:,2])
)
ax1.set_title('Dendrogramme Hiérarchique des Confessions Reddit', fontsize=16, pad=20)
ax1.set_xlabel('Clusters', fontsize=12)
ax1.set_ylabel('Distance', fontsize=12)

# Sous-graphique 2: Heatmap des mots-clés
ax2 = fig.add_subplot(gs[1])

# Créer la heatmap avec les mots les plus fréquents par cluster
im = ax2.imshow(cluster_word_matrix, cmap='YlOrRd', aspect='auto')

# Configurer les labels
ax2.set_xticks(range(n_top_words))
ax2.set_xticklabels([f"Mot {i+1}" for i in range(n_top_words)], rotation=45, ha='right')
ax2.set_yticks(range(optimal_clusters))
ax2.set_yticklabels([f"Cluster {i+1}" for i in range(optimal_clusters)])

# Ajouter une colorbar
plt.colorbar(im, ax=ax2, shrink=0.8, label='Score TF-IDF')

ax2.set_title('Mots Caractéristiques\npar Cluster', fontsize=14, pad=20)
ax2.set_xlabel('Top Mots', fontsize=12)

# Ajout des valeurs dans la heatmap
for i in range(optimal_clusters):
    for j in range(n_top_words):
        if cluster_word_matrix[i, j] > 0:
            text = ax2.text(j, i, f'{cluster_word_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('dendrogramme_1.png', dpi=300, bbox_inches='tight')
print("Dendrogramme complet créé et enregistré comme 'dendrogramme_1.png'")

# 5. CRÉATION D'UNE VERSION ALTERNATIVE AVEC LES VRAIS MOTS AFFICHÉS
print("\nÉtape 5: Création d'une version avec les mots réels affichés")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12), 
                               gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.1})

# Dendrogramme
dend = dendrogram(
    Z,
    labels=[f"Cluster_{cluster}" for cluster in final_clusters],
    leaf_rotation=90,
    leaf_font_size=10,
    ax=ax1,
    color_threshold=0.7*max(Z[:,2])
)
ax1.set_title('Dendrogramme Hiérarchique', fontsize=16, pad=20)
ax1.set_xlabel('Clusters', fontsize=12)
ax1.set_ylabel('Distance', fontsize=12)

# Table des mots-clés à droite
ax2.axis('off')
ax2.set_title('Mots Caractéristiques par Cluster', fontsize=16, pad=20)

# Créer un texte formaté avec les mots-clés
y_pos = 0.95
for cluster_id in range(1, optimal_clusters + 1):
    if cluster_id in top_words_per_cluster:
        # Titre du cluster
        ax2.text(0.05, y_pos, f'CLUSTER {cluster_id}:', 
                fontsize=14, fontweight='bold', transform=ax2.transAxes)
        y_pos -= 0.05
        
        # Mots du cluster
        words_text = []
        for word, score in top_words_per_cluster[cluster_id][:8]:  # Top 8 mots
            words_text.append(f'• {word} ({score:.3f})')
        
        words_str = '\n'.join(words_text)
        ax2.text(0.1, y_pos, words_str, 
                fontsize=10, transform=ax2.transAxes, verticalalignment='top')
        
        y_pos -= 0.15  # Espace entre clusters

plt.savefig('dendrogramme_2.png', dpi=300, bbox_inches='tight')
print("Version alternative créée et enregistrée comme 'dendrogramme_2.png'")

# 6. ANALYSE DÉTAILLÉE DES CLUSTERS (non supervisée)
print("\nÉtape 6: Analyse automatique des clusters")

with open('analyse_clusters_complete.txt', 'w', encoding='utf-8') as f:
    f.write("ANALYSE AUTOMATIQUE DES CLUSTERS DE CONFESSIONS REDDIT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Nombre de clusters déterminé automatiquement: {optimal_clusters}\n\n")
    
    for cluster_id in range(1, optimal_clusters + 1):
        f.write(f"\nCLUSTER {cluster_id}:\n")
        f.write("-" * 30 + "\n")
        
        cluster_mask = sample_df['cluster'] == cluster_id
        cluster_confessions = sample_df[cluster_mask]
        
        f.write(f"Nombre de confessions: {len(cluster_confessions)}\n\n")
        
        f.write("Mots caractéristiques:\n")
        if cluster_id in top_words_per_cluster:
            for word, score in top_words_per_cluster[cluster_id]:
                f.write(f"  {word}: {score:.4f}\n")
        
        f.write("\nTitres des confessions:\n")
        for _, row in cluster_confessions.iterrows():
            f.write(f"  - {row['title']}\n")
        
        f.write("\nExemple de confession:\n")
        if len(cluster_confessions) > 0:
            example = cluster_confessions.iloc[0]
            text = example['selftext']
            if len(text) > 300:
                text = text[:300] + "..."
            f.write(f"{text}\n")

print("Analyse complète terminée et enregistrée dans 'analyse_clusters_complete.txt'")

# 7. VISUALISATION t-SNE DES CLUSTERS AUTOMATIQUES
print("\nÉtape 7: Visualisation t-SNE des clusters automatiques")

tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, sample_size-1))
X_tsne = tsne.fit_transform(X_dense)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=final_clusters, 
                     cmap='tab10', 
                     s=100, 
                     alpha=0.8,
                     edgecolors='w')

# Ajouter les étiquettes des clusters
for cluster_id in range(1, optimal_clusters + 1):
    cluster_points = X_tsne[final_clusters == cluster_id]
    if len(cluster_points) > 0:
        centroid = cluster_points.mean(axis=0)
        plt.annotate(f'Cluster {cluster_id}', 
                    xy=(centroid[0], centroid[1]),
                    fontsize=12, 
                    fontweight='bold',
                    color='black',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

plt.title('Visualisation t-SNE des Clusters Automatiques', fontsize=16)
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('tsne_clusters.png', dpi=300, bbox_inches='tight')
print("Visualisation t-SNE créée et enregistrée comme 'tsne_clusters_automatiques.png'")

print("\n" + "="*70)
print("ANALYSE TERMINÉE AVEC SUCCÈS!")
print("="*70)
print("Fichiers générés:")
print("  - dendrogramme_complet_avec_mots.png: Dendrogramme avec heatmap")
print("  - dendrogramme_avec_mots_texte.png: Dendrogramme avec mots en texte")
print("  - analyse_clusters_complete.txt: Analyse détaillée automatique")
print("  - tsne_clusters_automatiques.png: Visualisation t-SNE")
print("\nCaractéristiques du clustering automatique:")
print(f"  - {optimal_clusters} clusters déterminés automatiquement")
print("  - Aucune supervision manuelle")
print("  - Basé sur la minimisation de la distance intra-cluster")
print("\nVous pouvez maintenant utiliser ces visualisations pour votre présentation!")