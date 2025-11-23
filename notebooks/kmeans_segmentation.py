# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: proba
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# %%
df = pd.read_csv("C:/projet/customer-segmentation/data/RFM.csv") 

# %%
df.head(10)

# %%
df.shape

# %%
df.columns

# %%
df.dtypes

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
df.dropna(subset=['Nom'], inplace=True)

# %%
X = df[['R', 'F', 'M']]

# %%
#standarisation
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(df[['R', 'F', 'M']])

# %%
##elbow find k
K_range = range(2, 13)
wcss = []   ##點到質心平方和
models = []
for k in K_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(rfm_scaled) ##訓練標準化之後的數據
    wcss.append(km.inertia_) ##存儲計算之後的質心
    models.append(km)##存儲每個k值下訓練好的模型

plt.figure()
plt.plot(list(K_range), wcss, marker='o')##x軸，y軸，點樣式
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("WCSS (inertia)")
plt.xticks(list(K_range))
plt.show()


# %%
best_k = 5
km = KMeans(n_clusters=best_k, n_init=50, random_state=42)
km.fit(rfm_scaled) ##訓練標準化之後的數據
labels = km.fit_predict(rfm_scaled)
df['cluster'] = labels

# %%
sil = silhouette_score(rfm_scaled, labels)
print(f"silhouette = {sil:.3f}  | sizes = {np.bincount(labels)}") ##輪廓係數評價

# %%
##centers 給出中心
centers_scaled = km.cluster_centers_
centers = pd.DataFrame(
    scaler.inverse_transform(centers_scaled),  # 用你前面的 StandardScaler
    columns=['R','F','M']
).assign(cluster=range(best_k))
print(centers.sort_values('cluster'))

# %%
profile = df.groupby('cluster').agg(
    n=('cluster','size'),
    R_mean=('R','mean'), F_mean=('F','mean'), M_mean=('M','mean'),
    R_median=('R','median'), F_median=('F','median'), M_median=('M','median')
).sort_values('n', ascending=False)
print(profile)

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

Z = PCA(n_components=2, random_state=42).fit_transform(rfm_scaled)
plt.figure()
plt.scatter(Z[:,0], Z[:,1], c=labels)  # 简单看一下簇分布
plt.title(f"KMeans K={best_k} (PCA 2D)")
plt.show()


# %%
## Reduce standardized features (rfm_scaled) to 2 dimensions with PCA for visualization 用离散色盘，避免颜色渐变误导
plt.scatter(Z[:,0], Z[:,1], c=labels, cmap='tab10', s=35, alpha=0.9)

##  Plot each sample in the 2D PCA space; color by its KMeans-assigned cluster label轻微抖动，减少重叠（只为展示）
eps = 0.03
plt.scatter(Z[:,0]+np.random.randn(len(Z))*eps,
            Z[:,1]+np.random.randn(len(Z))*eps,
            c=labels, cmap='tab10', s=20, alpha=0.6)


# %%
##Quick diagnostics (cluster sizes, silhouette, PCA variance)
import numpy as np
from sklearn.metrics import silhouette_score
print("sizes:", np.bincount(labels))                 # 每簇样本数
print("silhouette:", silhouette_score(rfm_scaled, labels))  # 分群质量（>0.25 基本可用）

from sklearn.decomposition import PCA
pca = PCA(n_components=3, random_state=42).fit(rfm_scaled)
print("PCA var (PC1,PC2,PC3):", pca.explained_variance_ratio_,
      "sum(PC1+PC2)=", pca.explained_variance_ratio_[:2].sum())
# 如果 PC1+PC2 < 0.6，2D 投影损失信息较多，本来就不容易“看出5簇”


# %%
##PCA 2D scatter with discrete colors + centroid markers + legend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

cmap = ListedColormap(['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd'])  # 5 个离散颜色

pca2 = PCA(n_components=2, random_state=42).fit(rfm_scaled)
Z = pca2.transform(rfm_scaled)

plt.figure()
plt.scatter(Z[:,0], Z[:,1], c=labels, cmap=cmap, s=35, alpha=0.9, edgecolors='k', linewidths=0.2)

# 画出簇中心（投影到同一 PCA 空间）
centers_2d = pca2.transform(km.cluster_centers_)
plt.scatter(centers_2d[:,0], centers_2d[:,1], c=range(km.n_clusters), cmap=cmap,
            marker='X', s=220, edgecolors='k', linewidths=1.0)
for i,(x,y) in enumerate(centers_2d):
    plt.text(x, y, str(i), ha='center', va='center', fontsize=10, weight='bold')

# 简单图例
handles = [plt.Line2D([0],[0], marker='o', color='w', label=f'Cluster {i}',
                      markerfacecolor=cmap(i), markeredgecolor='k', markersize=8)
           for i in range(km.n_clusters)]
plt.legend(handles=handles, frameon=False, ncol=3)

plt.title(f"KMeans K={km.n_clusters} (PCA 2D)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()


# %%
##Decision regions (cluster boundaries) in the PCA 2D plane
import numpy as np
xx, yy = np.meshgrid(np.linspace(Z[:,0].min()-0.5, Z[:,0].max()+0.5, 400),
                     np.linspace(Z[:,1].min()-0.5, Z[:,1].max()+0.5, 400))
grid_pca = np.c_[xx.ravel(), yy.ravel()]
# 把 PCA 平面上的网格点“反投影”回标准化特征空间，再用 kmeans 预测簇
grid_scaled = pca2.inverse_transform(grid_pca)
pred = km.predict(grid_scaled).reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, pred, levels=np.arange(km.n_clusters+1)-0.5, alpha=0.15, cmap=cmap)
plt.scatter(Z[:,0], Z[:,1], c=labels, cmap=cmap, s=25, edgecolors='k', linewidths=0.2)
plt.scatter(centers_2d[:,0], centers_2d[:,1], c=range(km.n_clusters), cmap=cmap,
            marker='X', s=220, edgecolors='k')
plt.title("Decision regions in PCA 2D"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()


# %%
##3D PCA scatter (keeps more structure than 2D)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
Z3 = PCA(n_components=3, random_state=42).fit_transform(rfm_scaled)
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=labels, cmap=cmap, s=20)
ax.set_title("PCA 3D"); plt.show()


# %%
df.to_csv("clustered_output.csv", index=False)
centers.to_csv("cluster_centers.csv", index=False)
profile.to_csv("cluster_profile.csv", index=False)

