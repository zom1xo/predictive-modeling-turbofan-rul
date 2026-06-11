# Project 6: Turbofan Engine Remaining Useful Life (RUL) Prediction

**Company:** UCT  
**Domain:** Predictive Maintenance / Machine Learning  
**Dataset:** NASA CMAPSS FD001  

---

## 1. Background

Predictive maintenance is a critical application of machine learning in aerospace and manufacturing. The ability to forecast when a component will fail allows operators to schedule maintenance proactively — reducing downtime, preventing catastrophic failures, and optimizing costs. Turbofan engines are complex systems with multiple sensors monitoring temperature, pressure, vibration, and other parameters. As an engine degrades, these sensor readings drift from their normal ranges, providing signals that can be used to estimate remaining operational life.

---

## 2. Problem Statement

The objective is to predict the **Remaining Useful Life (RUL)** — the number of operational cycles remaining before failure — for a fleet of turbofan engines using historical sensor data. The training set contains run-to-failure trajectories from 100 engines under a single operating condition (Sea Level) and a single fault mode (HPC Degradation). The test set contains trajectories that end some time before failure, and the goal is to predict how many cycles remain after the last recorded data point.

This is a **regression problem** with time-series characteristics. The target variable (RUL) is continuous and must be predicted for each cycle of each engine.

---

## 3. Design / Approach

The project followed a structured machine learning pipeline:

1. **Exploratory Data Analysis (EDA):**  
   - Loaded and inspected the FD001 training dataset (100 engines, 20,631 rows, 26 columns).  
   - Visualized all 21 sensor readings over time for individual engines to identify degradation trends.  
   - Computed the RUL target variable as `max_cycles - current_cycle` for each engine.

2. **Feature Engineering:**  
   - Created rolling mean and standard deviation features (window = 10 cycles) for all 21 sensors to capture short-term behavior.  
   - Created rolling trend (slope) features to capture the direction and rate of sensor drift.  
   - This expanded the feature set from 21 raw sensors to 63 engineered features.  
   - Dropped raw sensor columns, keeping only engineered features along with operational settings.

3. **Train/Test Split:**  
   - Split by engine (80% train, 20% test) rather than randomly, ensuring the model is evaluated on completely unseen engines. This prevents data leakage.

4. **Modeling:**  
   - **Baseline:** Linear Regression — a simple, interpretable model to establish a performance floor.  
   - **Primary Model:** Random Forest Regressor (100 trees, max depth = 10) — a non-linear ensemble method well-suited to capturing complex sensor interactions.

5. **Evaluation:**  
   - Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score.  
   - Compared models using bar charts and a predicted-vs-actual scatter plot.  
   - Analyzed feature importance to identify which engineered signals most strongly predict RUL.

---

## 4. Implementation Details

- **Language:** Python 3  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Environment:** Google Colab  
- **Dataset:** NASA CMAPSS FD001 (train_FD001.txt)  
- **Feature engineering:**  
  - Rolling mean (window=10) → 21 features  
  - Rolling standard deviation (window=10) → 21 features  
  - Rolling trend/slope (window=10) → 21 features  
- **Scaling:** StandardScaler applied after train/test split  
- **Random Forest parameters:** n_estimators=100, max_depth=10, random_state=42  

---

## 5. Results

| Model            | RMSE (cycles) | MAE (cycles) | R²    |
|------------------|---------------|--------------|-------|
| Linear Regression | 35.80         | 28.13        | 0.703 |
| Random Forest     | 34.08         | 24.39        | 0.731 |

- Random Forest reduced prediction error by **1.72 cycles (RMSE)** compared to the Linear Regression baseline.  
- R² improved from **0.703 to 0.731**, indicating the model explains 73.1% of the variance in RUL. 
- The top predictive features were rolling means and trends of sensors that exhibited visible degradation during EDA (particularly `sensor_4_roll_mean` and `sensor_9_trend`).  
- Prediction accuracy is highest when the engine is close to failure (low RUL) and decreases for early-life predictions — a known characteristic of RUL models.

---

## 6. Learnings

- **Time-series handling:** For predictive maintenance problems, splitting data by unit (engine) rather than randomly is essential to avoid data leakage and overly optimistic results.  
- **Feature engineering matters:** Raw sensor values alone were insufficient; rolling statistics and trend features dramatically improved model performance.  
- **Interpretability:** Random Forest feature importance provided insights into which sensors are most indicative of degradation — valuable for domain experts.  
- **Real-world applicability:** The project demonstrated how machine learning can be applied to industrial sensor data to reduce maintenance costs and prevent unplanned downtime.  
- **Next steps:** The model could be further improved by incorporating additional datasets (FD002–FD004), experimenting with deep learning approaches (LSTM), and building a live monitoring dashboard.

---

**Repository:** [(https://github.com/zom1xo/predictive-modeling-turbofan-rul)]  
**Notebook:** `Project6_Turbofan_EDA.ipynb`
