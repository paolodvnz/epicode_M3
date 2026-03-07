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
