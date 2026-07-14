# Predictive Modeling Using Machine Learning: Employee Attrition Prediction

## Project Objective
The goal of this project is to predict employee attrition (**Yes/No**) based on demographic and work-related features using supervised machine learning classification algorithms.

## Dataset Overview
* **Source:** `employee_attrition_data.csv`
* **Size:** 400 records, 10 columns
* **Target Variable:** `attrition` (Yes: 207, No: 193)
* **Features Used:** 
  * *Numerical:* `age`, `income`, `years_experience`, `hours_worked_per_week`, `satisfaction_score`, `distance_from_office_km`
  * *Categorical:* `education`, `department` (Processed via One-Hot Encoding)

## Workflow & Methodology
1. **Data Exploration & Preprocessing:** Evaluated dataset structure, handled basic statistics, dropped irrelevant identifiers (`employee_id`), and encoded categorical variables.
2. **Train-Test Split:** Partitioned data into an **80% training set** (320 records) to train models and a **20% test set** (80 records) for unseen evaluation.
3. **Model Training:** Implemented and compared three classification algorithms:
   * Decision Tree Classifier
   * Random Forest Classifier
   * Logistic Regression (evaluated with and without `StandardScaler` feature scaling)
4. **Evaluation:** Performance was assessed using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC/AUC curves.

## Key Findings & Results

| Model | Accuracy |
| :--- | :---: |
| **Decision Tree** | **86.25%** |
| **Random Forest** | **86.25%** |
| Logistic Regression (Scaled) | 82.50% |
| Logistic Regression (Unscaled) | 81.25% |

### Core Insights
* **Top Performers:** The **Decision Tree** and **Random Forest** models tied for the highest overall accuracy at **86.25%**.
* **Feature Sensitivity:** Logistic Regression showed a convergence warning and lower performance with unscaled data. Applying `StandardScaler` improved its accuracy from 81.25% to 82.50%, confirming its sensitivity to feature scales compared to tree-based models.

## Limitations & Future Scope
* **Data Scale:** The dataset is relatively small (400 records) and synthetic, meaning results may not fully capture complex real-world attrition dynamics.
* **Hyperparameter Tuning:** Models were evaluated using default configuration settings; future iterations could leverage `GridSearchCV` or `RandomizedSearchCV` to optimize hyperparameters and boost predictive performance further.

## Requirements & Libraries
* Python 3.x
* Pandas
* Matplotlib
* Scikit-Learn
