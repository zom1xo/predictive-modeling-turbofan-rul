# Turbofan Engine RUL Prediction

Predicting Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS FD001 dataset. Built during UCT machine learning internship.

## Overview
- **Problem:** Predict how many operational cycles remain before engine failure using 21 sensor measurements.
- **Dataset:** FD001 (100 engines, single operating condition, single fault mode).
- **Approach:** EDA → Feature Engineering (rolling statistics, trends) → Linear Regression baseline → Random Forest.

## Results
| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 35.80 | 28.13 | 0.703 |
| Random Forest | 34.08 | 24.39 | 0.731 |

## Files
- `Project6_Turbofan_EDA.ipynb` — Full notebook: data loading, EDA, feature engineering, modeling, evaluation.
- `Project_Report.md` — Detailed project report with background, methodology, results, and learnings.

## Tools
Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Google Colab
