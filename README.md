# EPICODE Modulo 3 — Machine Learning

Progetti didattici sul Machine Learning.
Ogni progetto è un Jupyter Notebook autonomo con analisi, visualizzazioni e commenti esplicativi.

---

## Progetti

### 1. Clustering con K-Means e DBSCAN

**File:** [`project_1_m3.ipynb`]

Esplorazione del dataset **Iris** tramite due algoritmi di clustering non supervisionato, con confronto dei risultati nel piano PCA 2D.

**Pipeline:**
1. Caricamento ed esplorazione del dataset (statistiche descrittive, pair plot)
2. Preprocessing — standardizzazione con `StandardScaler`
3. **DBSCAN** — clustering density-based con analisi della sensibilità al parametro `eps`
4. **K-Means** — selezione del k ottimale tramite Elbow Method e Silhouette Score
5. Visualizzazione PCA 2D — cluster predetti vs specie reali

**Librerie:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

### 2. Regressione con PCA e Modelli Lineari

**File:** [`project_2_m3.ipynb`]

Esplorazione del dataset **Linnerud** tramite modelli di regressione lineare con riduzione dimensionale PCA, su un dataset multi-output di misurazioni fisiologiche e prestazioni fisiche.

**Pipeline:**
1. Caricamento ed esplorazione del dataset (statistiche descrittive, visualizzazione 3D)
2. Preprocessing — standardizzazione con `StandardScaler` e riduzione dimensionale con `PCA`
3. **Addestramento** di `LinearRegression`, `Lasso` e `Ridge` su due configurazioni di target: PC1 del target e variabile singola (`Waist`)
4. Analisi comparativa dei risultati (RMSE, R²) e test dell'impatto della standardizzazione pre-PCA
5. Ri-addestramento nello spazio ridotto **PC1 Features → PC1 Target** con visualizzazione delle rette di regressione

**Librerie:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`

---
