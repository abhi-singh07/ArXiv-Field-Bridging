# STAT 636 Project: Do Field Bridges Accelerate Scientific Impact?

Evidence from ArXiv Collaboration Networks

## 📊 Project Overview

This project investigates which types of field bridges lead to higher research impact in computational sciences using large-scale ArXiv collaboration networks (2010-2023).

**Research Question**: Do authors who bridge high-value field pairs achieve higher citation impact than specialists or those bridging unrelated areas?

## 👥 Team Members
- Abhishek Singh (UIN: 136005114)
- Joshua Vemana (UIN: 735007887)  
- Rushil Ravi (UIN: 836000314)

## 🎯 H1 Results: Field-Specific Bridging Premium

**Hypothesis**: Authors bridging CS-Statistics achieve highest citation advantage (~40%), followed by CS-Mathematics (~20%), while within-field bridging yields minimal gains.

### Key Findings:
- ✅ **CS-STAT bridges**: +55.5% citation advantage 
- ✅ **CS-MATH bridges**: +34.3% citation advantage  
- ✅ **Within-CS bridges**: +3.9% citation advantage
- 📊 **H1 Support**: Strong - All predictions confirmed

### Statistical Significance:
- All bridge type effects statistically significant (p < 0.001)
- **Sample size**: 440,591 authors
- **Model R²**: 0.290
- **Robust standard errors** (HC1)

### Bridge Type Effects (vs CS Specialists):
| Bridge Type | Citation Advantage | p-value |
|-------------|-------------------|---------|
| CS-STAT | +55.5% | < 0.001 |
| CS-MATH | +34.3% | < 0.001 |
| Within-CS | +3.9% | < 0.001 |

## 📁 Project Structure

ArXiv-Field-Bridging/
├── src/
│ ├── analysis/
│ │ ├── h1_analysis.py # Complete H1 analysis code
│ │ └── h2_analysis.py # H2 analysis (TODO)
│ └── preprocessing.py # Data preprocessing
├── results/
│ └── H1/
│ ├── h1_realistic_results.txt # Complete regression output
│ ├── h1_realistic_findings.json # Key metrics and counts
│ ├── h1_realistic_results.png # Visualizations
│ └── authors_h1_realistic.parquet # Analysis dataset
├── notebooks/
│ ├── h1_exploration.ipynb # H1 exploratory analysis
│ └── h2_exploration.ipynb # H2 exploratory analysis (TODO)
└── README.md