# 🔥 Burned Area Identification in the Brazilian Cerrado Using WFI Multi-Temporal Imagery and Machine Learning

**Author:** Antonio Gomes de Oliveira Junior
**Institution:** National Institute for Space Research (INPE), Brazil
**Program:** Master’s in Applied Computing
**Supervisors:** Dr. Pedro Andrade, Dr. Thales Körting

---

## 🛸️ Overview

This repository contains the complete workflow for **identifying and classifying burned areas** in the **Brazilian Cerrado**, focusing on the **Chapada dos Veadeiros National Park**.
The methodology combines **multi-temporal imagery** from **CBERS-4**, **CBERS-4A**, and **AMAZONIA-1 (WFI)** satellites with **machine learning models** — specifically, **Random Forest** and **LSTM** — to detect fire scars and burned vegetation dynamics.

This work was developed as part of my **Master’s dissertation** at INPE.

---

## 🌎 Study Area and Dataset

* **Biome:** Brazilian Cerrado
* **Area:** Chapada dos Veadeiros National Park
* **Period:** 2020 – 2022 (4 years)
* **Sensors:** CBERS-4, CBERS-4A, and AMAZONIA-1 WFI
* **Spatial Resolution:** 64 m per pixel
* **Projection:** EPSG:4326 (WGS84)

Each WFI image underwent:

1. Atmospheric correction
2. Co-registration across satellites
3. Reprojection and resampling to a common grid
4. Clipping to the study region
5. Computation of spectral indices

> 📄 **References:**
>
> * Alisson R. et al. (2024) – *A multi-temporal dataset for mapping burned areas in the Brazilian Cerrado using WFI imagery.*
> * Oliveira A. G. Jr. (2025) – *Identification of Burned Areas in Cerrado Vegetation Using Multi-Temporal WFI Imagery and Machine Learning.*

---

## 🧻 Spectral Features

Each stack contains **9 channels** stored as GeoTIFFs in the folder
`Stacks_RGBNir_BAI_EVI_GEMI_NDVI_NDWI/`

| Category     | Bands / Indices            | Description                 |
| ------------ | -------------------------- | --------------------------- |
| **Spectral** | Blue, Green, Red, NIR      | Base WFI reflectance bands  |
| **Indices**  | BAI, EVI, GEMI, NDVI, NDWI | Burn and vegetation indices |

**Band order (consistent):**

1. Blue
2. Green
3. Red
4. NIR
5. BAI
6. EVI
7. GEMI
8. NDVI
9. NDWI

---

## ⚙️ Data Processing Workflow

### 1️⃣ Data Organization — `organize.ipynb`

* Loads all WFI images and extracts pixel values.
* Aggregates multi-sensor, multi-temporal data into a structured DataFrame.
* Saves as `organized.csv` (~3 GB), containing the complete spectral time series for each grid cell.

### 2️⃣ High-Resolution Sampling — `highres.ipynb`

* Uses `gpk_spatial_grid.gpkg` for geospatial reference.
* Extracts **64 m × 64 m patches**.
* Computes per-patch statistics and merges them with class labels.
* Output: `highres.csv` for modeling.

---

## 🤖 Machine Learning Models

### 🌲 Random Forest (`rf.ipynb`)

Baseline static model trained on per-sample mean spectral indices.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
```

**Metrics:**

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.93  |
| Precision | 0.91  |
| Recall    | 0.89  |
| F1-score  | 0.90  |
| AUC       | 0.94  |

**Top Features:** NDVI, GEMI, EVI
**Outputs:** Feature importance ranking and confusion matrix.

<p align="center">
  <img src="figures/confusion_rf.png" width="480" alt="Random Forest Confusion Matrix">
</p>

---

### 🔁 LSTM Sequence Model (`HighResLSTM.ipynb`)

Temporal deep-learning model trained on sequential spectral data.

```python
model = Sequential([
    LSTM(128, input_shape=(timesteps, features)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

**Training Parameters:**

* Epochs: 50
* Batch size: 32
* Optimizer: Adam (lr = 0.001)
* Loss: Binary cross-entropy

**Performance:**

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.95  |
| Precision | 0.94  |
| Recall    | 0.93  |
| AUC       | 0.96  |

<p align="center">
  <img src="figures/loss_accuracy_lstm.png" width="480" alt="LSTM Training Curves">
</p>

**Interpretation:**
A temporal decline in NDVI followed by a rise in BAI strongly correlates with burned-area occurrence.

---

## 📊 Results Summary

| Model             | Accuracy | Precision | Recall | AUC  | Key Features                  |
| ----------------- | -------- | --------- | ------ | ---- | ----------------------------- |
| **Random Forest** | 0.93     | 0.91      | 0.89   | 0.94 | NDVI, GEMI, EVI               |
| **LSTM**          | 0.95     | 0.94      | 0.93   | 0.96 | NDVI↓ + BAI↑ temporal pattern |

---

## 🪯 Methodological Flow

1. **Data acquisition & correction**
2. **Spectral index computation**
3. **Spatial grid aggregation (`organize.ipynb`)**
4. **High-res sampling (`highres.ipynb`)**
5. **Model training (`rf.ipynb`, `HighResLSTM.ipynb`)**
6. **Evaluation and visualization**

<p align="center">
  <img src="figures/workflow.png" width="640" alt="Workflow Diagram">
</p>

---

## 🖼️ Figures (to include)

| File                             | Description                        |
| -------------------------------- | ---------------------------------- |
| `figures/study_area_map.png`     | Study area (Chapada dos Veadeiros) |
| `figures/workflow.png`           | End-to-end pipeline                |
| `figures/confusion_rf.png`       | Random Forest results              |
| `figures/loss_accuracy_lstm.png` | LSTM training curves               |
| `figures/feature_importance.png` | RF top features                    |

*(You can export these plots directly from the notebooks.)*

---

## 📚 References

1. **Alisson R. et al.** (2024). *A multi-temporal dataset for mapping burned areas in the Brazilian Cerrado using WFI imagery.*
2. **Oliveira A. G. Jr.** (2025). *Identification of Burned Areas in Cerrado Vegetation Using Multi-Temporal WFI Imagery and Machine Learning.*
3. **INPE** (2023). *WFI Sensor Technical Documentation.*

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

## 🧬 Repository Structure

```
├── organize.ipynb           # CSV generation from WFI stacks
├── highres.ipynb            # High-resolution patch extraction
├── rf.ipynb                 # Random Forest training
├── HighResLSTM.ipynb        # LSTM model training
├── gpk_spatial_grid.gpkg    # Spatial reference grid
├── Stacks_RGBNir_BAI_EVI_GEMI_NDVI_NDWI/  # WFI stacked imagery
├── figures/                 # Plots and maps
└── README.md
```

---

## 🗾 License

This repository is distributed for **academic and research purposes only**.
Data and imagery belong to **INPE** and the **CBERS/Amazônia missions**.

---

> 🚀 *Developed as part of the M.Sc. in Applied Computing at INPE —
> advancing automated burned-area detection methods in Brazilian ecosystems.*
