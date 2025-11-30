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

## 🎯 H2 Results: Optimal Diversity (The Inverted U-Shape)

**Hypothesis**: The relationship between field diversity and impact follows an inverted U-shape—authors active in 2–3 related fields reach the highest citation impact, while both specialists and extreme generalists underperform.

### Key Findings:
- ✅ **Inverted U-Shape Confirmed**: Significant negative quadratic term indicates diminishing returns for extreme generalists.
- ✅ **The "Sweet Spot"**: Peak impact occurs at **Shannon Entropy ≈ 1.06**, which corresponds to active work in **2-3 related fields**.
- 📉 **Diminishing Returns**: Impact declines significantly as authors spread themselves too thin across unrelated fields.

### Statistical Significance:
- **Quadratic Term ($\beta_2$)**: -0.777 (p < 0.001)
- **Linear Term ($\beta_1$)**: +1.641 (p < 0.001)
- **Model Fit ($R^2$)**: 0.474 (Strong explanatory power)

---

## 🛡️ Robustness Checks

To validate our findings against over-dispersed count data, we utilized **Negative Binomial Regression**, the "gold standard" for citation analysis.

### 1. Validating H1 (Bridge Hierarchy)
The hierarchy of bridge value holds firm even under rigorous testing. The **Incidence Rate Ratios (IRR)** confirm that CS-Statistics bridges provide the largest multiplicative boost to citation counts.

| Bridge Type | IRR (Multiplicative Effect) | Interpretation |
|-------------|----------------------------|----------------|
| **CS-STAT** | **1.136** | **+13.6% Citations** (Highest) |
| CS-MATH | 1.118 | +11.8% Citations |
| Within-CS | 1.014 | +1.4% Citations (Negligible) |

*(Note: All effects significant at p < 0.001)*

### 2. Validating H2 (Optimal Diversity)
The "Inverted U-Shape" remains robust. The quadratic term for Shannon Entropy squared (`entropy_sq`) remained **negative (-0.154)** and **highly significant (p < 0.001)** in the Negative Binomial model. This confirms that the "diminishing returns" of excessive diversity is a real structural phenomenon, not an artifact of the linear model.

---

## 📈 Visualizations

The project generated the following key visualizations to support these findings:

1.  **`results/H1/h1_realistic_results.png`**: Bar charts showing the raw citation premium and percentage advantages of CS-STAT authors compared to others.
2.  **`results/H2/h2_diversity_curve.png`**: A scatter plot with a quadratic regression curve (red line) clearly illustrating the rise and fall of impact as diversity increases.
3.  **`results/H2/h2_diversity_bins.png`**: A binned bar chart showing that "Mid-Diversity" authors outperform both "Low" (Specialist) and "High" (Generalist) groups.

---

## 🚀 Conclusion & Implications

Our study moves beyond the generic advice that "interdisciplinarity is good." We provide quantitative evidence that **strategic bridging** is the key to accelerating scientific impact.

* **Theoretical Contribution**: We refine Structural Hole theory by showing that not all bridges are equal—their value depends on the **complementarity** of the fields (CS & Statistics > CS & AI).
* **Practical Advice**: Researchers should aim for **moderate diversity** (Mastery of 2 distinct fields) rather than shallow generalism. The most valuable profile in computational science today is the **CS-Statistician bridge**.