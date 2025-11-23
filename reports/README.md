# Customer Segmentation Report (RFM + K-Means)

> One-page deliverable for stakeholders. Replace the placeholders in `<>` if needed.

## 1) Executive Summary
- **Objective:** Segment customers using RFM to support targeting and retention.
- **Chosen K:** `<5>` (selected via Elbow; clear bend at K≈5).
- **Overall Quality:** Silhouette = `<0.xx>`; Cluster sizes = `<[n0, n1, n2, n3, n4]>`.
- **Key Takeaways:**  
  - C0 `<name>`: `<short insight>`  
  - C1 `<name>`: `<short insight>`  
  - C2 `<name>`: `<short insight>`  
  - C3 `<name>`: `<short insight>`  
  - C4 `<name>`: `<short insight>`  

## 2) Data
- **Columns:** R (recency, days), F (frequency), M (monetary).
- **Preprocessing:** optional `log1p` on F/M (if highly skewed), then `StandardScaler`.
- **Sample size:** `<N>` rows after cleaning.

## 3) Method
- **K-Means** with `n_init=20–50`, `random_state=42` on standardized features.
- **Model selection:** Elbow method on WCSS (see *Figures → elbow.png*).
- **Validation:** Silhouette score + cluster size distribution.
- **Visualization:** PCA 2D scatter with centroids (see *Figures → pca_k5.png*).

## 4) Results
### 4.1 Cluster sizes
Insert the output of `np.bincount(labels)` here.

### 4.2 Cluster profiles (mean/median)
See `cluster_profile.csv`. Typical reading:
- **Low R** = recent; **High F** = frequent; **High M** = valuable.
- Name clusters using business-friendly labels, e.g. *High-Value Loyal*, *Potential*, *Dormant*, *Price-Sensitive*, *New/Trial*.

### 4.3 Centroids (original scale)
See `cluster_centers.csv`. These are inverse-transformed R/F/M means per cluster.

## 5) Figures
- `figures/elbow.png` – Elbow method curve (WCSS vs K).
- `figures/pca_k5.png` – PCA 2D scatter colored by cluster with centroid markers.
- *(Optional)* `figures/cluster_sizes.png` – Bar chart of sizes.

## 6) Reproducibility
- Fixed seeds: `random_state=42`, `n_init>=20`.
- Use the same scaler for scoring (save/load with `joblib`).

## 7) Next Steps
- Monitor cluster shifts monthly and design actions per segment:
  - **High-Value Loyal:** loyalty perks, early access.
  - **Potential:** nurture flows, incentives to increase F/M.
  - **Dormant:** reactivation campaigns.
  - **Price-Sensitive:** bundles/coupons.
  - **New/Trial:** onboarding and first-to-second purchase push.
