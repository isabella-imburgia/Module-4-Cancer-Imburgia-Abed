# %%
import pandas as pd # Pandas
import numpy as np # Numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.decomposition import PCA

data = pd.read_csv(
    '/Users/zain/Documents/GitHub/Module-4-Cancer-Imburgia-Abed/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    #'/Users/ajq2af/OneDrive - University of Virginia\Documents/UVA/BME 2315/Module-4-Cancer-Imburgia-Abed/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0, header=0
)
metadata_df = pd.read_csv(
    '/Users/zain/Documents/GitHub/Module-4-Cancer-Imburgia-Abed/data/TRAINING_SET_GSE62944_metadata.csv',
    #'/Users/ajq2af/OneDrive - University of Virginia/Documents/UVA/BME 2315/Module-4-Cancer-Imburgia-Abed/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0, header=0
)

# ── Explore the data ───────────────────────────────────────────────────────────
print(data.shape)
print(data.info())
print(data.describe())
 
# ── Explore the metadata ───────────────────────────────────────────────────────
print(metadata_df.info())
print(metadata_df.describe())
 
# ── Subset data for a specific cancer type ─────────────────────────────────────
cancer_type = 'UCEC'
 
# Strip whitespace to avoid silent mismatches
data.columns = data.columns.str.strip()
metadata_df.index = metadata_df.index.str.strip()
 
# Get sample IDs for the target cancer type
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
common_samples = data.columns.intersection(cancer_samples)
print(cancer_samples)
 
# Subset the main data to include only UCEC samples
UCEC_data = data[common_samples]


#  ── Gene subsetting ────────────────────────────────────────────────────────────
desired_gene_list = ['CASP1','CASP2','CASP3','CASP5','CASP6','CASP7','CASP8','CASP9','CASP10','BCL2','BCL2L1','BCL2L2','BCL2L10','BCL2A1','BCL2L11','BAX','BAK1','BAD','BID','BBC3','BNIP3','BNIP3L','MCL1','TNF','FAS','FADD','TRADD','TNFRSF1A','TNFRSF1B','TNFSF10','TNFRSF10A','TNFRSF10B','TNFRSF10C','TNFRSF10D','CFLAR','NFKB1','RELA','NFKBIA','IKBKB','CHUK','TRAF2','TRAF6','MAP3K7','TAB1','AKT1','AKT2','AKT3','PIK3CA','PIK3R1','PTEN','MTOR','GSK3B','RICTOR','SOS1','TP53','MDM2','MDM4','ATM','CHEK1','CHEK2','CDKN1A','CDKN2A','RB1','EP300','MAPK1','MAPK3','MAPK8','MAPK9','MAPK10','MAPK12','MAPK13','MAPK14','RAF1','BRAF','MAP2K1','MAP2K2','MAP2K3','MAP2K4','MAP2K7','STAT1','STAT3','STAT5A','JAK1','JAK2','JAK3','SOCS3','IL6R','XIAP','BIRC2','BIRC3','BIRC5','APAF1','CYCS','DIABLO','AIFM1','RIPK1','RIPK2','BCL10','TP73','HIF1A','MAPKAPK2']
gene_list = [g for g in desired_gene_list if g in UCEC_data.index]
 
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] subsets by index label; .iloc[] subsets by integer position
UCEC_gene_data = UCEC_data.loc[gene_list]  # uses validated gene_list (safe)
print(UCEC_gene_data.head())


# ── Basic statistics on subsetted gene data ────────────────────────────────────
print(UCEC_gene_data.describe())
print(UCEC_gene_data.var(axis=1))     # Variance of each gene across samples
print(UCEC_gene_data.mean(axis=1))    # Mean expression per gene
print(UCEC_gene_data.median(axis=1))  # Median expression per gene

# ── Survival time summary for UCEC ────────────────────────────────────────────
metadata_df['OS.time'] = pd.to_numeric(metadata_df['OS.time'], errors='coerce')
filtered_df = metadata_df[metadata_df['cancer_type'] == cancer_type]
avg_OStime = filtered_df['OS.time'].mean()
print(f"Average OS time for {cancer_type}: {avg_OStime:.2f}")
 
# ── Merge expression data with metadata ───────────────────────────────────────
# Transpose so rows = samples, columns = genes
gene_df = UCEC_gene_data.T
UCEC_metadata = metadata_df.loc[common_samples]  # align to common_samples
UCEC_merged = gene_df.merge(UCEC_metadata, left_index=True, right_index=True)
print(UCEC_merged.head())

# ── Plotting ───────────────────────────────────────────────────────────────────
# Boxplot: histological diagnosis vs. OS time (Seaborn)
plt.figure()
sns.boxplot(
    data=UCEC_merged,
    x='histologic_diagnosis',
    y='OS.time'
)
plt.title(f"OS Time by Histologic Diagnosis ({cancer_type})")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()
 
# Boxplot: TP53 and AKT1 expression in UCEC samples (Pandas)
available_genes = [g for g in ['TP53', 'AKT1'] if g in UCEC_merged.columns]
UCEC_merged[available_genes].plot.box()
plt.title(f"TP53 and AKT1 Expression in {cancer_type} Samples")
plt.tight_layout()
plt.show()

# ── Prepare samples × genes matrix and attach metadata ───────────────────────
# Transpose so rows = samples, columns = genes
df = UCEC_gene_data.T.copy()  # shape: samples × genes

# Merge histologic_diagnosis from metadata into df
df = df.join(metadata_df[['histologic_diagnosis']], how='left')
# ── Remove samples with missing histologic_diagnosis ─────────────────────────  <-- ADD HERE
df = df.dropna(subset=['histologic_diagnosis'])
df = df[df['histologic_diagnosis'].str.strip() != '']

# ── PCA ───────────────────────────────────────────────────────────────────────
X = df[gene_list].values  # samples × genes, numeric only

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

explained = pca.explained_variance_ratio_ * 100  # percentage

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['histologic_diagnosis'],
    palette="Set2",
    s=100
)
plt.title("PCA of UCEC Gene Expression")
plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
plt.legend(title="Histologic Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# umap 
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.patches as mpatches

scaled_data = StandardScaler().fit_transform(X)

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(scaled_data)

# ── Color by histologic_diagnosis (matches your PCA plot) ─────────────────────
diagnoses = df['histologic_diagnosis']
categories = diagnoses.unique()
palette = sns.color_palette("Set2", len(categories))
color_map = dict(zip(categories, palette))
colors = diagnoses.map(color_map)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=colors, s=100, alpha=0.8)

# Legend
handles = [mpatches.Patch(color=color_map[d], label=d) for d in categories]
plt.legend(handles=handles, title="Histologic Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP of UCEC Gene Expression")
plt.tight_layout()
plt.show()

# clustering with KMeans
model = KMeans(n_clusters=3, random_state=0)
model.fit(X)
y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="Set2", s=100)
plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
plt.title("KMeans Clustering of UCEC Gene Expression")
plt.tight_layout()
plt.show()
# DBSCAN
# %%
from sklearn.cluster import DBSCAN

# Run on PCA coordinates (2D) instead of raw X
dbscan = DBSCAN(eps=1.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_pca)   # <-- X_pca not X

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_dbscan, cmap="Set2", s=100)
plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
plt.title("DBSCAN Clustering of UCEC Gene Expression")
plt.tight_layout()
plt.show()

print(f"Clusters found: {len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)}")
print(f"Noise points (label = -1): {(y_dbscan == -1).sum()}")

## main learning model
# %%
# %%
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# %%
# Use your already-loaded UCEC merged data
# X = gene expression features, y = histologic diagnosis labels

# ── Filter to binary classification (Endometrioid vs Serous) ──────────────────
binary_df = UCEC_merged[
    UCEC_merged['histologic_diagnosis'].isin([
        'Endometrioid endometrial adenocarcinoma',
        'Serous endometrial adenocarcinoma'
    ])
].copy()

# Only keep genes that are actually present in binary_df columns
available_genes = [g for g in gene_list if g in binary_df.columns]

# Encode labels: Endometrioid = 0, Serous = 1
le = LabelEncoder()
y = le.fit_transform(binary_df['histologic_diagnosis'])
y_label = list(binary_df['histologic_diagnosis'])
print(f"Samples — Endometrioid: {(y==0).sum()}, Serous: {(y==1).sum()}")

# ── Feature matrix: all 100 genes ─────────────────────────────────────────────
X_all = binary_df[available_genes].values

# ── PCA for visualization (replaces TP53 vs AKT1 scatter) ────────────────────
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X_all)
explained_vis = pca_vis.explained_variance_ratio_ * 100

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_vis[:, 0],
    y=X_vis[:, 1],
    hue=y_label,
    palette="Set1",
    s=100,
    edgecolors='k',
    alpha=0.8
)
plt.xlabel(f"PC1 ({explained_vis[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_vis[1]:.1f}% variance)")
plt.title("PCA of All 100 Genes: Endometrioid vs Serous (UCEC)")
plt.legend(title="Histologic Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ── Logistic Regression on all 100 genes ──────────────────────────────────────
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

lr_model = LogisticRegression(penalty=None, max_iter=1000).fit(X_scaled, y)
print(f"Logistic Regression Accuracy (all genes): {lr_model.score(X_scaled, y):.3f}")

# Plot decision boundary projected onto PC1/PC2
xx, yy = np.meshgrid(
    np.linspace(X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1, 300),
    np.linspace(X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1, 300)
)

# Project grid back to gene space via inverse PCA, then scale, then predict
grid_gene_space = pca_vis.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
grid_scaled = scaler.transform(grid_gene_space)
Z = lr_model.decision_function(grid_scaled).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.6)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
sns.scatterplot(
    x=X_vis[:, 0],
    y=X_vis[:, 1],
    hue=y_label,
    edgecolors='k',
    palette="Set1",
    alpha=0.8
)
plt.xlabel(f"PC1 ({explained_vis[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_vis[1]:.1f}% variance)")
plt.title("Logistic Regression Decision Boundary (All 100 Genes)\nEndometrioid vs Serous — projected onto PCA")
plt.legend(title="Histologic Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ── Decision Tree on all 100 genes ────────────────────────────────────────────
dt_model = DecisionTreeClassifier(max_depth=3).fit(X_all, y)
print(f"Decision Tree Accuracy (all genes): {dt_model.score(X_all, y):.3f}")

plt.figure(figsize=(16, 6))
plot_tree(
    dt_model,
    feature_names=available_genes,
    class_names=le.classes_,
    filled=True,
    fontsize=9
)
plt.title("Decision Tree: Classifying UCEC Subtypes (All 100 Genes)")
plt.tight_layout()
plt.show()

# ── Top features by importance (Decision Tree) ────────────────────────────────
importances = pd.Series(dt_model.feature_importances_, index=available_genes)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 5))
top_features.plot(kind='bar', color='steelblue')
plt.title("Top 15 Gene Importances (Decision Tree)")
plt.ylabel("Feature Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# %% TRY BETTER FEATURES — loop over gene pairs to find best classifier
from itertools import combinations
from sklearn.model_selection import cross_val_score

best_score = 0
best_pair = None

for gene1, gene2 in combinations(gene_list, 2):
    if gene1 in binary_df.columns and gene2 in binary_df.columns:
        X_pair = binary_df[[gene1, gene2]].values
        score = cross_val_score(
            LogisticRegression(penalty=None, max_iter=1000),
            X_pair, y, cv=5
        ).mean()
        if score > best_score:
            best_score = score
            best_pair = (gene1, gene2)

print(f"Best gene pair: {best_pair[0]} and {best_pair[1]}")
print(f"Cross-validated accuracy: {best_score:.3f}")