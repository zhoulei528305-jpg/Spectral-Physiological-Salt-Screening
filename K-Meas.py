import os
os.environ["OMP_NUM_THREADS"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator  # pip install kneed

# ==============================================================================
# 1. 读取数据
# ==============================================================================
file_path = r""

df = pd.read_excel(file_path, index_col=0)
df.index = df.index.astype(str).str.strip()

# ==============================================================================
# 2. 四分位法剔除异常值（对每一列单独处理）
# ==============================================================================
df_clean = df.copy()

for col in df_clean.columns:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean[col] = df_clean[col].where(
        (df_clean[col] >= lower) & (df_clean[col] <= upper), other=np.nan
    )

n_before = len(df_clean)
df_clean = df_clean.dropna()
n_after = len(df_clean)
print(f"✅ 异常值剔除完成：原始 {n_before} 个品种 → 剔除后 {n_after} 个品种（去掉了 {n_before - n_after} 个）")

# ==============================================================================
# 3. 提取最后一列用于聚类
# ==============================================================================
col_name = df_clean.columns[-1]
y = df_clean.iloc[:, -1].values.reshape(-1, 1)

# ==============================================================================
# 4. 肘图计算（k 范围 1~15）
# ==============================================================================
K_range = range(1, 16)
inertias = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(y)
    inertias.append(km.inertia_)

# ==============================================================================
# 5. 自动检测 Elbow 点
# ==============================================================================
kl = KneeLocator(
    list(K_range),
    inertias,
    curve="convex",
    direction="decreasing"
)
elbow_k = kl.elbow

if elbow_k is None:
    print("⚠️ 未能自动检测到明显拐点，默认使用 k=3")
    elbow_k = 3
else:
    print(f"✅ 自动检测 Elbow 点：k = {elbow_k}")

# ==============================================================================
# 6. 用 elbow_k 做最终聚类
# ==============================================================================
km_final = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
labels = km_final.fit_predict(y)
centers = km_final.cluster_centers_.flatten()

# 按聚类中心从小到大重新排序标签（便于语义一致，如：类0=低，类1=中，类2=高）
sorted_idx = np.argsort(centers)
remap = {old: new for new, old in enumerate(sorted_idx)}
labels = np.array([remap[l] for l in labels])
centers = np.sort(centers)

df_clean = df_clean.copy()
df_clean['Cluster'] = labels

# 打印各簇样本数
for c in range(elbow_k):
    print(f"  簇 {c}：{(labels == c).sum()} 个品种，中心值 = {centers[c]:.4f}")

# ==============================================================================
# 7. 绘图：肘图 + 散点图（左右并排）
# ==============================================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 28
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

colors = ['#4C8FC0', '#E07B54', '#5BAD72', '#A868B0',
          '#D4A84B', '#D95F5F', '#7DB8A4', '#C97FB5']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
label_size = 32
tick_size = 28
title_size = 34

# ── 左图：肘图 ──────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.plot(list(K_range), inertias,
         color='steelblue', linewidth=2.5,
         marker='o', markersize=9,
         markerfacecolor='white', markeredgewidth=2.5,
         markeredgecolor='steelblue')

elbow_inertia = inertias[elbow_k - 1]
ax1.annotate(f'Elbow (k={elbow_k})',
             xy=(elbow_k, elbow_inertia),
             xytext=(elbow_k + 1.5, elbow_inertia + (max(inertias) * 0.05)),
             fontsize=26,
             color='black',
             arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
ax1.axvline(x=elbow_k, color='black', linestyle='--', linewidth=1.5, alpha=0.6)

ax1.set_xlabel("Number of Clusters (k)", fontsize=label_size, color='black')
ax1.set_ylabel("Inertia (SSE)", fontsize=label_size, color='black')
ax1.set_xticks(list(K_range))
ax1.tick_params(axis='both', labelsize=tick_size, colors='black')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_edgecolor('black')
ax1.spines['bottom'].set_edgecolor('black')
ax1.set_title("Elbow Method", fontsize=title_size, color='black')

# ── 右图：K-Means 聚类散点图 ────────────────────────────────────────────────
ax2 = axes[1]
x_vals = np.arange(len(y))  # 样本序号作为x轴

for c in range(elbow_k):
    mask = labels == c
    ax2.scatter(x_vals[mask], y[mask].flatten(),
                color=colors[c % len(colors)],
                label=f'Cluster {c}  (n={mask.sum()})',
                s=60, alpha=0.85, edgecolors='none')
    # 标注聚类中心水平线
    ax2.axhline(y=centers[c], color=colors[c % len(colors)],
                linestyle='--', linewidth=1.5, alpha=0.7)

ax2.set_xlabel("Sample Index", fontsize=label_size, color='black')
ax2.set_ylabel(col_name, fontsize=label_size, color='black')
ax2.tick_params(axis='both', labelsize=tick_size, colors='black')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_edgecolor('black')
ax2.spines['bottom'].set_edgecolor('black')
ax2.legend(fontsize=26, frameon=False, labelcolor='black')
ax2.set_title(f"K-Means Clustering (k={elbow_k})", fontsize=title_size, color='black')

plt.tight_layout()

# ==============================================================================
# 8. 保存图像 (TIFF)
# ==============================================================================
output_fig = file_path.replace(".xlsx", "_Elbow_KMeans.tiff")
plt.savefig(output_fig, dpi=600, format='tiff',
            bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
print(f"✅ 图像已保存: {output_fig}")
plt.show()

# ==============================================================================
# 9. 保存聚类结果到 Excel
# ==============================================================================
output_excel = file_path.replace(".xlsx", "_KMeans_result.xlsx")
df_clean.to_excel(output_excel)
print(f"✅ 聚类结果已保存: {output_excel}")