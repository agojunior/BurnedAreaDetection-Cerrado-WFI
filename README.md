# 🔥 Burned Area Identification in the Brazilian Cerrado Using WFI Multi-Temporal Imagery and Machine Learning

**Author:** Antonio Gomes de Oliveira Junior  
**Institution:** National Institute for Space Research (INPE), Brazil  
**Program:** Master’s in Applied Computing  
**Supervisors:** Dr. Pedro Andrade, Dr. Thales Körting  

---

## 🛰️ Overview

This repository contains the complete workflow for **identifying and classifying burned areas** in the **Brazilian Cerrado**, focusing on the **Chapada dos Veadeiros National Park**.  
The methodology combines **multi-temporal imagery** from **CBERS-4**, **CBERS-4A**, and **AMAZONIA-1 (WFI)** satellites with **machine learning models** — specifically, **Random Forest** and **LSTM** — to detect fire scars and burned vegetation dynamics.

This work was developed as part of my **Master’s dissertation** at INPE.

---

## 🌎 Study Area and Dataset

- **Biome:** Brazilian Cerrado  
- **Area:** Chapada dos Veadeiros National Park  
- **Period:** 2020 – 2022 (4 years)  
- **Sensors:** CBERS-4, CBERS-4A, and AMAZONIA-1 WFI  
- **Spatial Resolution(s):**  
  - **500 m × 500 m** aggregated grid (**organized.csv**)  
  - **64 m × 64 m** high‑resolution patches (**highres.csv**)  
- **Projection:** EPSG:4326 (WGS84)

Each WFI image underwent:
1. Atmospheric correction  
2. Co-registration across satellites  
3. Reprojection and resampling to a common grid  
4. Clipping to the study region  
5. Computation of spectral indices

> 📄 **References:**  
> - Alisson R. et al. (2024) – *A multi-temporal dataset for mapping burned areas in the Brazilian Cerrado using WFI imagery.*  
> - Oliveira A. G. Jr. (2025) – *Identification of Burned Areas in Cerrado Vegetation Using Multi-Temporal WFI Imagery and Machine Learning.*

---

## 🧮 Spectral Features

Each stack contains **9 channels** stored as GeoTIFFs in the folder  
`Stacks_RGBNir_BAI_EVI_GEMI_NDVI_NDWI/`

| Category | Bands / Indices | Description |
|-----------|----------------|--------------|
| **Spectral** | Blue, Green, Red, NIR | Base WFI reflectance bands |
| **Indices** | BAI, EVI, GEMI, NDVI, NDWI | Burn and vegetation indices |

**Assumed band order (consistent across stacks):**
1. Blue • 2. Green • 3. Red • 4. NIR • 5. BAI • 6. EVI • 7. GEMI • 8. NDVI • 9. NDWI

---

## ⚙️ Data Processing Workflow

###  Data Organization — `organize.ipynb`
- Loads all WFI images and extracts pixel values.  
- Aggregates multi-sensor, multi-temporal data into a structured DataFrame.  
- Saves as **`organized.csv`** (≈3 GB), containing the complete spectral time series for each **500 m** grid cell.

### High-Resolution Sampling — highres.ipynb

This notebook refines the dataset by mapping each grid cell of the Chapada dos Veadeiros National Park into multiple 64 m × 64 m patches.
The gpk_spatial_grid.gpkg file provides the polygon grid used to overlay and extract reflectance and spectral indices from every available WFI scene.

Processing steps:

Geospatial extraction – For each polygon cell in the GeoPackage, pixel values from the 9 bands (Blue → NDWI) are read using rasterio.mask.

Statistical summarization – Each patch is summarized by descriptive statistics (mean, median, standard deviation) per band, yielding one feature vector per patch.

Temporal alignment – Observations from different dates are joined using the acquisition timestamp in the filename, maintaining time-series consistency.

Label merging – Patches inherit the corresponding label (nb, pb, tb) from the master label table, later simplified to binary:

0 = not / partially burned

1 = completely burned.

Export – The resulting dataset (highres.csv) contains thousands of samples representing spatially distinct 64 m patches across multiple years.


# 🤖 Models & Results (Separated by Resolution)

Below we separate the experiments for **both resolutions** and **both model types**.

## A) 500 m × 500 m — *organized.csv*

### A1. 🌲 Random Forest (RF-500m)
**Notebook:** `rf.ipynb` (baseline on 500 m aggregated features)  
**Setup:**
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
```
**Results (example from notebook):**
| Metric | Value |
|---|---|
| Accuracy |  0.97 |
| Precision | ~0.91 |
| Recall | ~0.82 |
| F1‑score | ~0.86 |

- **Top features:** NIR, BAI
- **Artifacts:** `500m_rf_confusion.png`, `500m_rf_importance.png`

> *Place your exported figures here:*  
> `<img src="500m_rf_confusion.png" width="480">`  
> `<img src="500m_rf_importance.png" width="480">`

---

## B) 64 m × 64 m — *highres.csv*

### B1. 🌲 Random Forest (RF-64m)
**Notebook:** `RFHighREs.ipynb` (RF on high‑resolution features)  
**Setup:** same hyper‑params as RF‑500m unless otherwise noted.  
**Results (example from notebook):**
| Metric              | Value  |
| ------------------- | ------ |
| **Accuracy**        | 0.9897 |
| **Precision (avg)** | 0.9878 |
| **Recall (avg)**    | 0.9167 |
| **F1-Score (avg)**  | 0.9491 |

- **Artifacts:** `64m_rf_confusion.png`, `64m_rf_importance.png`

> *Place your exported figures here:*  
> `<img src="64m_rf_confusion.png" width="480">`  
> `<img src="64m_rf_importance.png" width="480">`

---

### B2. 🔁 LSTM (LSTM-64m)

**Architecture (typical):**
```python
model = Sequential([
    LSTM(128, input_shape=(timesteps, features)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```
**Training:** 50 epochs, batch 32, Adam(1e‑3), BCE loss.  
**Notebook:** `HighResLSTM.ipynb` (sequence model at 64 m)  
**Results (example from notebook):**
| Metric              | Value  |
| ------------------- | ------ |
| **Accuracy**        | 0.9844 |
| **Precision (avg)** | 0.8972 |
| **Recall (avg)**    | 0.9844 |
| **F1-Score (avg)**  | 0.9356 |

- **Artifacts:** `64m_lstm_loss_acc.png`, `64m_lstm_confusion.png`

> *Place your exported figures here:*  
> `<img src="64m_lstm_loss_acc.png" width="480">`  
> `<img src="64m_lstm_confusion.png" width="480">`

---

## 🧭 Methodological Flow

1. **Data acquisition & correction**  
2. **Spectral index computation**  
3. **Spatial grid aggregation (`organize.ipynb`)**  
4. **High‑res sampling (`highres.ipynb`)**  
5. **Model training**  
   - RF‑500m → `rf.ipynb`  
   - RF‑64m → `RFHighREs.ipynb`  
   - LSTM‑64m → `HighResLSTM.ipynb`  
6. **Evaluation and visualization**

---

## 🧠 Citation

If you use this repository or dataset, please cite:

```bibtex
@mastersthesis{oliveira2025burnedareas,
  author  = {Antonio Gomes de Oliveira Junior},
  title   = {Identification of Burned Areas in the Brazilian Cerrado Using WFI Multi-Temporal Imagery and Machine Learning},
  school  = {INPE – National Institute for Space Research},
  year    = {2025}
}
```

---

## 🗂 Repository Structure

```
├── organize.ipynb                 # CSV generation from WFI stacks (500 m)
├── highres.ipynb                  # High‑resolution patch extraction (64 m)
├── rf.ipynb                       # Random Forest (500 m)
├── RFHighREs.ipynb                # Random Forest (64 m)
├── HighResLSTM.ipynb              # LSTM (64 m)
├── <LSTM‑organized>.ipynb         # (optional) LSTM (500 m) – add link/name if different
├── gpk_spatial_grid.gpkg          # Spatial reference grid
├── Stacks_RGBNir_BAI_EVI_GEMI_NDVI_NDWI/  # WFI stacks
├──                        # Exported plots
└── README.md
```

---

## 🧾 License

This repository is distributed for **academic and research purposes only**.  
Data and imagery belong to **INPE** and the **CBERS/Amazônia missions**.

---

> 🛰️ *Developed as part of the M.Sc. in Applied Computing at INPE —  
> with the goal of advancing automated burned‑area detection methods in Brazilian ecosystems.*
