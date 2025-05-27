import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    if text is None or text in ["[removed]", "[deleted]"]:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    stop_words = {"i", "me", "my", "we", "our", "you", "your", "he", "him", "his", 
                  "she", "her", "it", "its", "they", "them", "their", "what", "which", 
                  "who", "this", "that", "these", "those", "am", "is", "are", "was", 
                  "were", "be", "been", "being", "have", "has", "had", "do", "does", 
                  "did", "a", "an", "the", "and", "but", "if", "or", "because", "as", 
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", 
                  "between", "into", "through", "during", "before", "after", "above", 
                  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
                  "under", "again", "further", "then", "once", "here", "there", "when", 
                  "where", "why", "how", "all", "any", "both", "each", "few", "more", 
                  "most", "other", "some", "such", "no", "nor", "not", "only", "own", 
                  "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
                  "don", "should", "now"}
    return ' '.join([word for word in tokens if word not in stop_words])

def create_dendrogram_with_legend(Z, clusters, top_words, n_clusters):
    cluster_names = {
        1: "Relations & Émotions", 2: "Famille & Parents", 3: "Vie Quotidienne",
        4: "Développement Personnel", 5: "Relations Sociales", 6: "Estime de Soi",
        7: "Problèmes de Sommeil & Stress"
    }
    
    colors = ['#FF6B6B', '#25201F', '#45B7D1', '#96CEB4', '#0209FF', '#DDA0DD', '#000000']
    set_link_color_palette(colors[:n_clusters])
    
    unique_labels = []
    seen = set()
    for cluster in clusters:
        if cluster not in seen:
            unique_labels.append(cluster_names.get(cluster, f'Cluster {cluster}'))
            seen.add(cluster)
        else:
            unique_labels.append('')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), 
                                   gridspec_kw={'width_ratios': [2.5, 1.5], 'wspace': 0.4})
    
    threshold = Z[-n_clusters+1, 2] if len(Z) >= n_clusters-1 else Z[-1, 2] * 0.7
    
    dend = dendrogram(Z, labels=unique_labels, leaf_rotation=90, leaf_font_size=9, ax=ax1,
                     color_threshold=threshold, above_threshold_color='lightgray')
    
    xlabels = ax1.get_xticklabels()
    for label in xlabels:
        text = label.get_text()
        if text in cluster_names.values():
            idx = list(cluster_names.values()).index(text)
            label.set_color(colors[idx])
            label.set_fontweight('bold')
    
    ax1.set_title('Classification Hiérarchique avec Légende', fontsize=16, pad=25)
    ax1.set_xlabel('Groupes Thématiques', fontsize=12, labelpad=30)
    ax1.set_ylabel('Distance de Dissimilarité', fontsize=12)
    ax1.axhline(y=threshold, color='red', linestyle=':', alpha=0.8, 
               label=f'Seuil ({n_clusters} clusters)')
    ax1.legend()
    
    # Légende
    ax2.axis('off')
    ax2.set_title('Caractéristiques des Groupes', fontsize=16, pad=25)
    
    y_pos = 0.95
    for cluster_id in range(1, n_clusters + 1):
        if cluster_id in top_words:
            color = colors[(cluster_id-1) % len(colors)]
            name = cluster_names.get(cluster_id, f'Cluster {cluster_id}')
            
            rect = plt.Rectangle((0.02, y_pos-0.05), 0.04, 0.06, 
                               facecolor=color, alpha=0.8, transform=ax2.transAxes)
            ax2.add_patch(rect)
            
            ax2.text(0.1, y_pos-0.02, name, fontsize=12, fontweight='bold', 
                    transform=ax2.transAxes)
            
            words = [word for word, score in top_words[cluster_id][:5]]
            words_str = ' • '.join(words)
            ax2.text(0.1, y_pos-0.04, f"• {words_str}", fontsize=9, 
                    transform=ax2.transAxes)
            
            y_pos -= 0.13
    
    plt.tight_layout()
    plt.savefig('dendrogram_legend.png', dpi=300, bbox_inches='tight')
    print("Dendrogramme avec légende créé: dendrogram_legend.png")

def create_dendrogram_with_heatmap(Z, clusters, top_words, n_clusters):
    cluster_names = {
        1: "Relations & Émotions", 2: "Famille & Parents", 3: "Vie Quotidienne",
        4: "Développement Personnel", 5: "Relations Sociales", 6: "Estime de Soi",
        7: "Problèmes de Sommeil & Stress"
    }
    
    colors = ['#FF6B6B', '#25201F', '#45B7D1', '#96CEB4', '#0209FF', '#DDA0DD', '#000000']
    set_link_color_palette(colors[:n_clusters])
    
    # Créer matrice des mots caractéristiques
    n_top_words = 10
    cluster_word_matrix = np.zeros((n_clusters, n_top_words))
    
    for cluster_id in range(1, n_clusters + 1):
        if cluster_id in top_words:
            top_scores = [score for word, score in top_words[cluster_id][:n_top_words]]
            cluster_word_matrix[cluster_id-1, :len(top_scores)] = top_scores
    
    unique_labels = []
    seen = set()
    for cluster in clusters:
        if cluster not in seen:
            unique_labels.append(cluster_names.get(cluster, f'Cluster {cluster}'))
            seen.add(cluster)
        else:
            unique_labels.append('')
    
    fig = plt.figure(figsize=(22, 12))
    ax1 = plt.subplot2grid((10, 10), (0, 0), colspan=7, rowspan=9)
    ax2 = plt.subplot2grid((10, 10), (0, 8), colspan=2, rowspan=7)
    
    threshold = Z[-n_clusters+1, 2] if len(Z) >= n_clusters-1 else Z[-1, 2] * 0.7
    
    dend = dendrogram(Z, labels=unique_labels, leaf_rotation=90, leaf_font_size=9, ax=ax1,
                     color_threshold=threshold, above_threshold_color='lightgray')
    
    xlabels = ax1.get_xticklabels()
    for label in xlabels:
        text = label.get_text()
        if text in cluster_names.values():
            idx = list(cluster_names.values()).index(text)
            label.set_color(colors[idx])
            label.set_fontweight('bold')
    
    ax1.set_title('Classification Hiérarchique avec Heatmap', fontsize=16, pad=25)
    ax1.set_xlabel('Groupes Thématiques', fontsize=12, labelpad=30)
    ax1.set_ylabel('Distance de Dissimilarité', fontsize=12)
    ax1.axhline(y=threshold, color='red', linestyle=':', alpha=0.8, 
               label=f'Seuil ({n_clusters} clusters)')
    ax1.legend()
    
    # Heatmap
    im = ax2.imshow(cluster_word_matrix, cmap='YlOrRd', aspect='auto')
    
    ax2.set_xticks(range(n_top_words))
    ax2.set_xticklabels([f"M{i+1}" for i in range(n_top_words)], 
                       rotation=90, fontsize=7)
    ax2.set_yticks(range(n_clusters))
    
    short_names = ["R&E", "F&P", "VQ", "DP", "RS", "ES", "PSS"]
    ax2.set_yticklabels(short_names[:n_clusters], fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax2, shrink=0.7, pad=0.2)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Score TF-IDF', fontsize=8)
    
    ax2.set_title('Intensité des Mots\nCaractéristiques', fontsize=12, pad=20)
    ax2.set_xlabel('Top Mots', fontsize=9, labelpad=15)
    
    plt.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.15, wspace=0.4, hspace=0.3)
    plt.savefig('dendrogram_heatmap.png', dpi=300, bbox_inches='tight')
    print("Dendrogramme avec heatmap créé: dendrogram_heatmap.png")

def create_tsne_plot(X, clusters, n_clusters):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    colors = ['#FF6B6B', '#25201F', '#45B7D1', '#96CEB4', '#0209FF', '#DDA0DD', '#000000']
    
    plt.figure(figsize=(10, 8))
    for cluster_id in range(1, n_clusters + 1):
        points = X_tsne[clusters == cluster_id]
        if len(points) > 0:
            plt.scatter(points[:, 0], points[:, 1], c=[colors[cluster_id-1]], 
                       s=100, alpha=0.7, label=f'Cluster {cluster_id}')
    
    plt.title('Visualisation t-SNE des Clusters', fontsize=14)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_clean.png', dpi=300, bbox_inches='tight')
    print("Visualisation t-SNE créée: tsne_clean.png")

def find_optimal_clusters(distance, max_clusters=8):
    Z = linkage(distance, method='ward')
    scores = []
    
    for n in range(2, max_clusters):
        labels = fcluster(Z, n, criterion='maxclust')
        cluster_distances = []
        for cluster_id in range(1, n + 1):
            indices = np.where(labels == cluster_id)[0]
            if len(indices) > 1:
                cluster_dist = distance[np.ix_(indices, indices)]
                cluster_distances.append(np.mean(cluster_dist))
        
        scores.append(np.mean(cluster_distances) if cluster_distances else float('inf'))
    
    return np.argmin(scores) + 2, Z

def analyze_clusters(df, X, vectorizer, clusters, n_clusters):
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    
    for cluster_id in range(1, n_clusters + 1):
        mask = df['cluster'] == cluster_id
        if mask.sum() > 0:
            indices = np.where(mask)[0]
            vectors = X[indices]
            mean_tfidf = np.asarray(vectors.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_words[cluster_id] = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
    
    with open('cluster_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(f"ANALYSE DES CLUSTERS ({n_clusters} groupes)\n")
        f.write("=" * 50 + "\n\n")
        
        for cluster_id in range(1, n_clusters + 1):
            mask = df['cluster'] == cluster_id
            cluster_data = df[mask]
            
            f.write(f"CLUSTER {cluster_id}:\n")
            f.write(f"Taille: {len(cluster_data)} confessions\n")
            
            if cluster_id in top_words:
                f.write("Mots-clés: ")
                words = [word for word, score in top_words[cluster_id][:5]]
                f.write(", ".join(words) + "\n")
            
            f.write("Exemples de titres:\n")
            for _, row in cluster_data.head(3).iterrows():
                f.write(f"  - {row['title']}\n")
            f.write("\n")
    
    print("Analyse sauvegardée: cluster_analysis.txt")
    return top_words

# EXECUTION PRINCIPALE
print("Clustering Hiérarchique de Confessions Reddit")

# 1. Chargement des données
dataset = load_dataset("SocialGrep/one-million-reddit-confessions")
df = pd.DataFrame(dataset['train'])
df = df[~df['selftext'].isin(['[removed]', '[deleted]', None, ''])]

sample_size = 50
sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)
sample_df['clean_text'] = sample_df['selftext'].apply(clean_text)

print(f"Échantillon: {sample_size} confessions")

# 2. Vectorisation
vectorizer = TfidfVectorizer(max_features=100, min_df=2, stop_words='english')
X = vectorizer.fit_transform(sample_df['clean_text'])
X_dense = X.toarray()

# 3. Clustering
similarity = cosine_similarity(X)
distance = 1 - similarity
optimal_clusters, Z = find_optimal_clusters(distance)
clusters = fcluster(Z, optimal_clusters, criterion='maxclust')
sample_df['cluster'] = clusters

print(f"Nombre optimal de clusters: {optimal_clusters}")

# 4. Analyse et visualisation
top_words = analyze_clusters(sample_df, X, vectorizer, clusters, optimal_clusters)
create_dendrogram_with_legend(Z, clusters, top_words, optimal_clusters)
create_dendrogram_with_heatmap(Z, clusters, top_words, optimal_clusters)
create_tsne_plot(X_dense, clusters, optimal_clusters)

print("\nAnalyse terminée!")
print("Fichiers générés:")
print("  - dendrogram_legend.png")
print("  - dendrogram_heatmap.png")
print("  - tsne_clean.png") 
print("  - cluster_analysis.txt")