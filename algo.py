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

def create_improved_dendrogram_with_heatmap(Z, final_clusters, top_words_per_cluster, optimal_clusters, sample_df):
    """
    Crée un dendrogramme amélioré avec heatmap des intensités comme l'original
    """
    
    # Définir des noms logiques pour les clusters basés sur l'analyse
    cluster_names = {
        1: "Relations & Émotions",
        2: "Famille & Parents", 
        3: "Vie Quotidienne",
        4: "Développement Personnel",
        5: "Relations Sociales",
        6: "Estime de Soi",
        7: "Problèmes de Sommeil & Stress"
    }
    
    # Palette de couleurs pour chaque cluster
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F0A500']
    
    # Extraire les mots caractéristiques pour chaque cluster
    n_top_words = 10
    cluster_word_matrix = np.zeros((optimal_clusters, n_top_words))
    
    for cluster_id in range(1, optimal_clusters + 1):
        if cluster_id in top_words_per_cluster:
            top_scores = [score for word, score in top_words_per_cluster[cluster_id][:n_top_words]]
            cluster_word_matrix[cluster_id-1, :len(top_scores)] = top_scores
    
    # Créer des labels uniques par cluster pour une meilleure lisibilité
    unique_labels = []
    cluster_seen = set()
    for cluster in final_clusters:
        if cluster not in cluster_seen:
            unique_labels.append(cluster_names.get(cluster, f'Cluster {cluster}'))
            cluster_seen.add(cluster)
        else:
            unique_labels.append('')  # Label vide pour les répétitions
    
    # Création de la figure avec espacement très augmenté
    fig = plt.figure(figsize=(26, 14))  # Figure encore plus grande
    
    # Utiliser des coordonnées manuelles pour un contrôle précis
    ax1 = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=9)  # Dendrogramme prend 70% largeur
    ax2 = plt.subplot2grid((10, 10), (0, 8), colspan=2, rowspan=7)  # Heatmap plus petite et décalée
    
    # Sous-graphique 1: Dendrogramme
    threshold = 0.8 * max(Z[:, 2])
    
    # Créer le dendrogramme avec couleurs automatiques
    dend = dendrogram(
        Z,
        labels=unique_labels,
        leaf_rotation=0,
        leaf_font_size=10,  # Police réduite
        ax=ax1,
        color_threshold=threshold,
        above_threshold_color='lightgray'
    )
    
    # Modifier les couleurs des branches
    lines = ax1.get_lines()
    for i, line in enumerate(lines):
        color_index = i % len(colors)
        line.set_color(colors[color_index])
        line.set_linewidth(2)
    
    # Colorer les labels avec plus d'espacement
    xlabels = ax1.get_xticklabels()
    cluster_color_map = {}
    for i, (cluster_id, name) in enumerate(cluster_names.items()):
        cluster_color_map[name] = colors[i % len(colors)]
    
    for label in xlabels:
        text = label.get_text()
        if text in cluster_color_map:
            label.set_color(cluster_color_map[text])
            label.set_fontweight('bold')
            label.set_fontsize(10)  # Police plus petite
    
    ax1.set_title('Classification Hiérarchique des Confessions Reddit', fontsize=16, pad=25)
    ax1.set_xlabel('Groupes Thématiques', fontsize=12, labelpad=25)  # Plus d'espace
    ax1.set_ylabel('Distance de Dissimilarité', fontsize=12)
    ax1.tick_params(axis='x', which='major', labelsize=9, pad=20)  # Plus d'espace sous les labels
    
    # Sous-graphique 2: Heatmap avec beaucoup plus d'espace
    im = ax2.imshow(cluster_word_matrix, cmap='YlOrRd', aspect='auto')
    
    # Labels de la heatmap avec police très réduite
    ax2.set_xticks(range(n_top_words))
    ax2.set_xticklabels([f"M{i+1}" for i in range(n_top_words)],  # "M1", "M2" au lieu de "Mot 1"
                       rotation=90, ha='center', fontsize=7)  # Rotation 90° et police 7
    ax2.set_yticks(range(optimal_clusters))
    
    # Noms de clusters abrégés pour la heatmap
    short_names = ["R&E", "F&P", "VQ", "DP", "RS", "ES", "PSS"]
    ax2.set_yticklabels(short_names[:optimal_clusters], fontsize=8)
    
    # Colorbar avec position ajustée
    cbar = plt.colorbar(im, ax=ax2, shrink=0.7, pad=0.2)  # Plus d'espace pour la colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Score TF-IDF', fontsize=8)
    
    ax2.set_title('Intensité des Mots\nCaractéristiques', fontsize=12, pad=20)
    ax2.set_xlabel('Top Mots', fontsize=9, labelpad=15)
    
    # Ajustement manuel des espacements pour éviter tout chevauchement
    plt.subplots_adjust(
        left=0.08,    # Marge gauche
        right=0.82,   # Marge droite (pour laisser place à la colorbar)
        top=0.92,     # Marge haute
        bottom=0.15,  # Marge basse (pour les labels du dendrogramme)
        wspace=0.4,   # Espace horizontal entre graphiques
        hspace=0.3    # Espace vertical
    )
    
    plt.savefig('dendrogramme_presentation_avec_intensite.png', dpi=300, bbox_inches='tight')
    print("Dendrogramme avec intensités créé: dendrogramme_presentation_avec_intensite.png")
    
    return fig

def create_improved_dendrogram_with_legend(Z, final_clusters, top_words_per_cluster, optimal_clusters, sample_df):
    """
    Crée un dendrogramme amélioré avec légende des mots (sans intensités)
    """
    
    cluster_names = {
        1: "Relations & Émotions",
        2: "Famille & Parents", 
        3: "Vie Quotidienne",
        4: "Développement Personnel",
        5: "Relations Sociales",
        6: "Estime de Soi",
        7: "Problèmes de Sommeil & Stress"
    }
    
    # Palette de couleurs distinctes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F0A500']
    cluster_colors = {i+1: colors[i % len(colors)] for i in range(optimal_clusters)}
    
    # Créer des labels uniques par cluster
    unique_labels = []
    cluster_seen = set()
    for cluster in final_clusters:
        if cluster not in cluster_seen:
            unique_labels.append(cluster_names.get(cluster, f'Cluster {cluster}'))
            cluster_seen.add(cluster)
        else:
            unique_labels.append('')  # Label vide pour les répétitions
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12), 
                                   gridspec_kw={'width_ratios': [3, 2], 'wspace': 0.1})
    
    # Dendrogramme avec couleurs et labels horizontaux
    dend = dendrogram(
        Z,
        labels=unique_labels,
        leaf_rotation=0,  # Horizontal
        leaf_font_size=11,
        ax=ax1,
        color_threshold=0.7*max(Z[:,2]),
        above_threshold_color='gray'
    )
    
    # Appliquer les couleurs aux labels
    xlabels = ax1.get_xticklabels()
    cluster_color_map = {
        "Relations & Émotions": colors[0],
        "Famille & Parents": colors[1], 
        "Vie Quotidienne": colors[2],
        "Développement Personnel": colors[3],
        "Relations Sociales": colors[4],
        "Estime de Soi": colors[5],
        "Problèmes de Sommeil & Stress": colors[6]
    }
    
    for label in xlabels:
        if label.get_text() in cluster_color_map:
            label.set_color(cluster_color_map[label.get_text()])
            label.set_fontweight('bold')
    
    ax1.set_title('Classification Hiérarchique des Confessions Reddit', fontsize=16, pad=20)
    ax1.set_xlabel('Groupes Thématiques', fontsize=12)
    ax1.set_ylabel('Distance de Dissimilarité', fontsize=12)
    
    # Améliorer l'espacement des labels horizontaux
    ax1.tick_params(axis='x', which='major', labelsize=11, pad=10)
    
    # Légende avec mots-clés
    ax2.axis('off')
    ax2.set_title('Caractéristiques des Groupes', fontsize=16, pad=20)
    
    y_pos = 0.95
    for cluster_id in range(1, optimal_clusters + 1):
        if cluster_id in top_words_per_cluster:
            color = cluster_colors[cluster_id]
            cluster_name = cluster_names.get(cluster_id, f'Cluster {cluster_id}')
            
            # Rectangle coloré
            rect = plt.Rectangle((0.02, y_pos-0.06), 0.05, 0.08, 
                               facecolor=color, alpha=0.8, transform=ax2.transAxes)
            ax2.add_patch(rect)
            
            # Titre du cluster
            ax2.text(0.12, y_pos-0.02, cluster_name, 
                    fontsize=14, fontweight='bold', transform=ax2.transAxes)
            
            # Mots du cluster (sans valeurs)
            words_text = []
            for word, score in top_words_per_cluster[cluster_id][:6]:
                words_text.append(f'• {word}')
            
            words_str = '\n'.join(words_text)
            ax2.text(0.12, y_pos-0.05, words_str, 
                    fontsize=10, transform=ax2.transAxes, verticalalignment='top')
            
            y_pos -= 0.13
    
    plt.savefig('dendrogramme_presentation_avec_legende.png', dpi=300, bbox_inches='tight')
    print("Dendrogramme avec légende créé: dendrogramme_presentation_avec_legende.png")
    
    return fig

def create_cluster_summary_table(top_words_per_cluster, final_clusters, sample_df, optimal_clusters):
    """
    Crée un tableau récapitulatif avec couleurs corrigées
    """
    cluster_names = {
        1: "Relations & Émotions",
        2: "Famille & Parents", 
        3: "Vie Quotidienne",
        4: "Développement Personnel",
        5: "Relations Sociales",
        6: "Estime de Soi",
        7: "Problèmes de Sommeil & Stress"
    }
    
    # Utiliser matplotlib directement pour les couleurs
    import matplotlib.colors as mcolors
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Couleurs en format matplotlib
    colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F0A500']
    
    # Données pour le tableau
    table_data = []
    cell_colors = []
    
    for cluster_id in range(1, optimal_clusters + 1):
        if cluster_id in top_words_per_cluster:
            cluster_mask = sample_df['cluster'] == cluster_id
            count = cluster_mask.sum()
            
            # Mots-clés principaux
            top_words = [word for word, score in top_words_per_cluster[cluster_id][:4]]
            keywords = ', '.join(top_words)
            
            table_data.append([
                cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
                f"{count} confessions",
                keywords
            ])
            
            # Couleurs pour cette ligne
            color = colors_list[(cluster_id-1) % len(colors_list)]
            # Convertir la couleur hex en RGB normalisé pour matplotlib
            color_rgb = mcolors.hex2color(color)
            # Créer différentes intensités pour les colonnes
            cell_colors.append([
                (*color_rgb, 0.7),  # Nom du cluster - plus opaque
                (*color_rgb, 0.3),  # Taille - moins opaque
                (*color_rgb, 0.3)   # Mots-clés - moins opaque
            ])
    
    # Créer le tableau
    table = ax.table(cellText=table_data,
                    colLabels=['Groupe Thématique', 'Taille', 'Mots-clés Principaux'],
                    cellLoc='left',
                    loc='center',
                    bbox=[0, 0, 1, 1],
                    cellColours=cell_colors)
    
    # Styliser le tableau
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Styliser l'en-tête
    header_color = '#2C3E50'
    for j in range(3):
        table[(0, j)].set_facecolor(header_color)
        table[(0, j)].set_text_props(weight='bold', color='white')
        table[(0, j)].set_fontsize(14)
    
    plt.title('Résumé des Groupes Thématiques Identifiés', 
              fontsize=18, fontweight='bold', pad=20)
    
    plt.savefig('tableau_clusters_presentation_couleurs.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("Tableau récapitulatif avec couleurs créé: tableau_clusters_presentation_couleurs.png")
    
    return fig

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

# Extraire les mots caractéristiques pour chaque cluster
feature_names = vectorizer.get_feature_names_out()
top_words_per_cluster = {}
n_top_words = 10

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
        
        top_words_per_cluster[cluster_id] = list(zip(top_words, top_scores))

# 4. CRÉATION DES DENDROGRAMMES AMÉLIORÉS
print("\nÉtape 4: Création des dendrogrammes améliorés")

# Version avec heatmap des intensités (comme l'original)
fig1 = create_improved_dendrogram_with_heatmap(Z, final_clusters, top_words_per_cluster, optimal_clusters, sample_df)

# Version avec légende colorée (sans intensités)
fig2 = create_improved_dendrogram_with_legend(Z, final_clusters, top_words_per_cluster, optimal_clusters, sample_df)

# Tableau récapitulatif avec couleurs corrigées
fig3 = create_cluster_summary_table(top_words_per_cluster, final_clusters, sample_df, optimal_clusters)

# 5. ANALYSE DÉTAILLÉE DES CLUSTERS
print("\nÉtape 5: Analyse automatique des clusters")

with open('analyse_clusters_complete.txt', 'w', encoding='utf-8') as f:
    f.write("ANALYSE AUTOMATIQUE DES CLUSTERS DE CONFESSIONS REDDIT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Nombre de clusters déterminé automatiquement: {optimal_clusters}\n\n")
    
    cluster_names = {
        1: "Relations & Émotions",
        2: "Famille & Parents", 
        3: "Vie Quotidienne",
        4: "Développement Personnel",
        5: "Relations Sociales",
        6: "Estime de Soi",
        7: "Problèmes de Sommeil & Stress"
    }
    
    for cluster_id in range(1, optimal_clusters + 1):
        f.write(f"\nCLUSTER {cluster_id} - {cluster_names.get(cluster_id, f'Cluster {cluster_id}')}:\n")
        f.write("-" * 50 + "\n")
        
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

# 6. VISUALISATION t-SNE DES CLUSTERS
print("\nÉtape 6: Visualisation t-SNE des clusters")

tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, sample_size-1))
X_tsne = tsne.fit_transform(X_dense)

plt.figure(figsize=(12, 8))

# Utiliser les mêmes couleurs
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F0A500']
color_map = {i+1: colors[i % len(colors)] for i in range(optimal_clusters)}

for cluster_id in range(1, optimal_clusters + 1):
    cluster_points = X_tsne[final_clusters == cluster_id]
    if len(cluster_points) > 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[color_map[cluster_id]], 
                   s=100, 
                   alpha=0.8,
                   edgecolors='w',
                   label=f'Cluster {cluster_id}')

# Ajouter les noms des clusters
cluster_names = {
    1: "Relations & Émotions",
    2: "Famille & Parents", 
    3: "Vie Quotidienne",
    4: "Développement Personnel",
    5: "Relations Sociales",
    6: "Estime de Soi",
    7: "Problèmes de Sommeil & Stress"
}

for cluster_id in range(1, optimal_clusters + 1):
    cluster_points = X_tsne[final_clusters == cluster_id]
    if len(cluster_points) > 0:
        centroid = cluster_points.mean(axis=0)
        plt.annotate(cluster_names.get(cluster_id, f'Cluster {cluster_id}'), 
                    xy=(centroid[0], centroid[1]),
                    fontsize=10, 
                    fontweight='bold',
                    color='black',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

plt.title('Visualisation t-SNE des Clusters Thématiques', fontsize=16, fontweight='bold')
plt.xlabel('Dimension t-SNE 1', fontsize=12)
plt.ylabel('Dimension t-SNE 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('tsne_clusters_presentation.png', dpi=300, bbox_inches='tight')
print("Visualisation t-SNE créée: tsne_clusters_presentation.png")

print("\n" + "="*70)
print("ANALYSE TERMINÉE AVEC SUCCÈS!")
print("="*70)
print("Fichiers générés:")
print("  - dendrogramme_presentation_avec_intensite.png: Avec heatmap d'intensité")
print("  - dendrogramme_presentation_avec_legende.png: Avec légende colorée")
print("  - tableau_clusters_presentation_couleurs.png: Tableau avec couleurs corrigées")
print("  - tsne_clusters_presentation.png: Visualisation t-SNE")
print("  - analyse_clusters_complete.txt: Analyse détaillée")
print("\nAméliorations apportées:")
print("  ✅ Conservation des degrés d'intensité (heatmap)")
print("  ✅ Couleurs corrigées dans le tableau")
print("  ✅ Noms logiques des clusters")
print("  ✅ Deux versions du dendrogramme (avec/sans intensités)")
print("  ✅ Interface professionnelle pour présentation")