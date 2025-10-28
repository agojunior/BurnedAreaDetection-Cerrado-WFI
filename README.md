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
- **Spatial Resolution:** 64 m per pixel  
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
- Loads all WFI images and extracts pixel values.  
- Aggregates multi-sensor, multi-temporal data into a structured DataFrame.  
- Saves as `organized.csv` (~3 GB), containing the complete spectral time series for each grid cell.

### 2️⃣ High-Resolution Sampling — `highres.ipynb`
- Uses `gpk_spatial_grid.gpkg` for geospatial reference.  
- Extracts **64 m × 64 m patches**.  
- Computes per-patch statistics and merges them with class labels.  
- Output: `highres.csv` for modeling.

---

## 🤖 Machine Learning Models

### 🌲 Random Forest (`rf.ipynb`)
Baseline static model trained on per-sample mean spectral indices.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
