# Clustering Hiérarchique de Confessions Reddit

Ce projet implémente une analyse de clustering hiérarchique sur des confessions Reddit en utilisant des techniques de traitement du langage naturel et d'apprentissage automatique non supervisé.

## 📋 Description

L'objectif de ce projet est d'analyser et de regrouper automatiquement des confessions Reddit en clusters thématiques cohérents. Le système utilise la vectorisation TF-IDF et le clustering hiérarchique pour identifier des groupes de confessions partageant des caractéristiques similaires.

## ✨ Fonctionnalités

- **Préprocessing avancé** : Nettoyage et normalisation des textes avec suppression des mots vides
- **Vectorisation TF-IDF** : Transformation des textes en représentations numériques
- **Clustering hiérarchique** : Regroupement automatique avec différentes méthodes de liaison
- **Visualisations interactives** : 
  - Dendrogrammes pour visualiser la hiérarchie des clusters
  - Projection t-SNE pour la visualisation en 2D
  - Comparaison des méthodes de liaison
- **Analyse des clusters** : Extraction des mots-clés caractéristiques de chaque groupe
- **Rapport détaillé** : Génération automatique d'un rapport d'analyse

## 🚀 Résultats

Le système identifie automatiquement plusieurs types de confessions :

### Cluster 0 - Problèmes pratiques et financiers
- **Mots-clés** : order, make, makes, happened, work
- **Thématiques** : Difficultés financières, problèmes de logement, coût de la vie

### Cluster 1 - Problèmes émotionnels et relationnels  
- **Mots-clés** : feel, like, know, dont, really, want, hate
- **Thématiques** : Relations amoureuses, estime de soi, problèmes familiaux, santé mentale

### Cluster 2 - Relations à long terme
- **Mots-clés** : relationship, long, time, wont
- **Thématiques** : Relations durables, engagement, confiance

### Cluster 3 - Observations quotidiennes
- **Thématiques** : Habitudes, comportements du quotidien

### Cluster 4 - Questionnements identitaires
- **Mots-clés** : mean, girl, years, dont
- **Thématiques** : Orientation sexuelle, identité, relations de genre

## 🛠️ Installation

### Prérequis

- Python 3.8+
- pip ou conda

### Installation des dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn datasets scipy
```

ou avec conda :

```bash
conda install pandas numpy matplotlib seaborn scikit-learn scipy
pip install datasets  # datasets n'est pas disponible via conda
```

### Dépendances complètes

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
datasets>=2.0.0
scipy>=1.7.0
```

## 📊 Utilisation

### Exécution de base

```bash
python algo.py
```

### Paramètres configurables

Dans le fichier `algo.py`, vous pouvez modifier :

- `sample_size` : Nombre de confessions à analyser (défaut: 50)
- `n_clusters` : Nombre de clusters souhaité (défaut: 5)
- `max_features` : Nombre maximum de caractéristiques TF-IDF (défaut: 100)

### Exemple d'utilisation personnalisée

```python
# Modifier la taille de l'échantillon
sample_size = 100

# Changer le nombre de clusters
n_clusters = 7

# Ajuster les paramètres TF-IDF
vectorizer = TfidfVectorizer(
    max_features=200, 
    min_df=3, 
    stop_words='english',
    ngram_range=(1, 2)  # Inclure les bigrammes
)

## 📈 Sorties générées

Le script génère automatiquement :

1. **dendrogram.png** : Dendrogramme principal montrant la hiérarchie des clusters
2. **linkage_methods.png** : Comparaison de 4 méthodes de liaison différentes
3. **optimal_clusters.png** : Analyse du nombre optimal de clusters
4. **tsne_clusters.png** : Visualisation 2D des clusters avec t-SNE
5. **cluster_analysis.txt** : Rapport détaillé avec mots-clés et exemples

## 🔧 Algorithmes utilisés

- **Préprocessing** : Tokenisation, suppression des mots vides, normalisation
- **Vectorisation** : TF-IDF (Term Frequency-Inverse Document Frequency)
- **Clustering** : Clustering hiérarchique agglomératif avec différentes méthodes de liaison
- **Visualisation** : t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Analyse** : Extraction des termes les plus discriminants par cluster

## 📚 Dataset

Le projet utilise le dataset "SocialGrep/one-million-reddit-confessions" disponible sur Hugging Face, contenant plus d'un million de confessions anonymes Reddit.

**Caractéristiques du dataset :**
- Source : r/confession et subreddits similaires
- Contenu : Texte des confessions + titres
- Taille : ~1M d'entrées
- Langue : Anglais principalement

## ⚙️ Configuration avancée

### Optimisation des performances

Pour de gros volumes de données :

```python
# Utiliser un échantillon stratifié
sample_df = df.sample(n=1000, random_state=42)

# Optimiser TF-IDF
vectorizer = TfidfVectorizer(
    max_features=500,
    min_df=5,
    max_df=0.95,
    stop_words='english'
)

# Utiliser PCA avant clustering pour réduire la dimensionnalité
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_dense)
```

### Méthodes de liaison disponibles

- **single** : Distance minimale entre clusters
- **complete** : Distance maximale entre clusters  
- **average** : Distance moyenne entre clusters
- **ward** : Minimise la variance intra-cluster (recommandée)

## 🔗 Références

- [Dataset Reddit Confessions](https://huggingface.co/datasets/SocialGrep/one-million-reddit-confessions)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [TF-IDF Vectorization](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
