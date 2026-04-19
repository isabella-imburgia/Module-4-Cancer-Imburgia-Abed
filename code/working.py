import pandas as pd # Pandas
import numpy as np # Numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.decomposition import PCA

data = pd.read_csv(
    '/Users/zain/Documents/GitHub/Module-4-Cancer-Imburgia-Abed/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0, header=0
)
metadata_df = pd.read_csv(
    '/Users/zain/Documents/GitHub/Module-4-Cancer-Imburgia-Abed/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0, header=0
)
 
# ── Explore the data ───────────────────────────────────────────────────────────
#print(data.shape)
#print(data.info())
#print(data.describe())
 
# ── Explore the metadata ───────────────────────────────────────────────────────
#print(metadata_df.info())
#print(metadata_df.describe())
 
# ── Subset data for a specific cancer type ─────────────────────────────────────
cancer_type = 'UCEC'
 
# Strip whitespace to avoid silent mismatches
data.columns = data.columns.str.strip()
metadata_df.index = metadata_df.index.str.strip()
 
# Get sample IDs for the target cancer type
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
common_samples = data.columns.intersection(cancer_samples)
#print(cancer_samples)
 
# Subset the main data to include only UCEC samples
UCEC_data = data[common_samples]

# %
#  ── Gene subsetting ────────────────────────────────────────────────────────────
desired_gene_list = ['CASP1','CASP2','CASP3','CASP5','CASP6','CASP7','CASP8','CASP9','CASP10','BCL2','BCL2L1','BCL2L2','BCL2L10','BCL2A1','BCL2L11','BAX','BAK1','BAD','BID','BBC3','BNIP3','BNIP3L','MCL1','TNF','FAS','FADD','TRADD','TNFRSF1A','TNFRSF1B','TNFSF10','TNFRSF10A','TNFRSF10B','TNFRSF10C','TNFRSF10D','CFLAR','NFKB1','RELA','NFKBIA','IKBKB','CHUK','TRAF2','TRAF6','MAP3K7','TAB1','AKT1','AKT2','AKT3','PIK3CA','PIK3R1','PTEN','MTOR','GSK3B','RICTOR','SOS1','TP53','MDM2','MDM4','ATM','CHEK1','CHEK2','CDKN1A','CDKN2A','RB1','EP300','MAPK1','MAPK3','MAPK8','MAPK9','MAPK10','MAPK12','MAPK13','MAPK14','RAF1','BRAF','MAP2K1','MAP2K2','MAP2K3','MAP2K4','MAP2K7','STAT1','STAT3','STAT5A','JAK1','JAK2','JAK3','SOCS3','IL6R','XIAP','BIRC2','BIRC3','BIRC5','APAF1','CYCS','DIABLO','AIFM1','RIPK1','RIPK2','BCL10','TP73','HIF1A','MAPKAPK2']
gene_list = [g for g in desired_gene_list if g in UCEC_data.index]
 
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] subsets by index label; .iloc[] subsets by integer position
UCEC_gene_data = UCEC_data.loc[gene_list]  # uses validated gene_list (safe)
#print(UCEC_gene_data.head())
 
# ── Basic statistics on subsetted gene data ────────────────────────────────────
#print(UCEC_gene_data.describe())
#print(UCEC_gene_data.var(axis=1))     # Variance of each gene across samples
#print(UCEC_gene_data.mean(axis=1))    # Mean expression per gene
#print(UCEC_gene_data.median(axis=1))  # Median expression per gene

# ── Survival time summary for UCEC ────────────────────────────────────────────
#metadata_df['OS.time'] = pd.to_numeric(metadata_df['OS.time'], errors='coerce')
#filtered_df = metadata_df[metadata_df['cancer_type'] == cancer_type]
#avg_OStime = filtered_df['OS.time'].mean()
#print(f"Average OS time for {cancer_type}: {avg_OStime:.2f}")
 
# ── Merge expression data with metadata ───────────────────────────────────────
# Transpose so rows = samples, columns = genes
#gene_df = UCEC_gene_data.T
#UCEC_metadata = metadata_df.loc[common_samples]  # align to common_samples
#UCEC_merged = gene_df.merge(UCEC_metadata, left_index=True, right_index=True)
#print(UCEC_merged.head())
 
# ── Plotting ───────────────────────────────────────────────────────────────────
# Boxplot: histological diagnosis vs. OS time (Seaborn)
#plt.figure()
#sns.boxplot(
#    data=UCEC_merged,
#    x='histologic_diagnosis',
#    y='OS.time'
#)
#plt.title(f"OS Time by Histologic Diagnosis ({cancer_type})")
#plt.xticks(rotation=30, ha='right')
#plt.tight_layout()
#plt.show()
 
# Boxplot: TP53 and AKT1 expression in UCEC samples (Pandas)
#available_genes = [g for g in ['TP53', 'AKT1'] if g in UCEC_merged.columns]
#UCEC_merged[available_genes].plot.box()
#plt.title(f"TP53 and AKT1 Expression in {cancer_type} Samples")
#plt.tight_layout()
#plt.show()

# ── Prepare samples × genes matrix and attach metadata ───────────────────────
# Transpose so rows = samples, columns = genes
df = UCEC_gene_data.T.copy()  # shape: samples × genes

# Merge histologic_diagnosis from metadata into df
df = df.join(metadata_df[['histologic_diagnosis']], how='left')

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