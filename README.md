# Predictive Maintenance Classification Project ⚙️

## Introduction

This project aims to predict potential machine failures based on sensor data from industrial equipment. The goal is to build a classification model that can identify machines likely to fail, enabling proactive maintenance scheduling and reducing downtime. This uses the **AI4I 2020 Predictive Maintenance Dataset**.

## Dataset

* **Source:** AI4I 2020 Predictive Maintenance Dataset (Commonly found on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) or Kaggle).
* **Description:** Contains synthetic data reflecting real-world predictive maintenance scenarios. Includes sensor readings like air temperature, process temperature, rotational speed, torque, and tool wear, along with machine type and failure information.
* **Key Characteristic:** The dataset is highly **imbalanced**, with only a small percentage of data points representing actual machine failures.

## Methodology

1.  **Exploratory Data Analysis (EDA):** (`eda.ipynb`)
    * Inspected data types, checked for missing values and duplicates.
    * Analyzed distributions of sensor readings using histograms and summary statistics.
    * Confirmed the significant target class imbalance.
    * Examined correlations between numerical features using a heatmap.
    * Compared sensor readings between failing and non-failing instances using box plots and grouped averages.
    * Investigated failure patterns across different machine types (`L`, `M`, `H`).

2.  **Data Preprocessing & Modeling:** (`model.ipynb`)
    * Selected relevant features (sensor readings, machine type), excluding identifiers and the detailed failure type to prevent target leakage.
    * Split the data into training and testing sets using **stratification** to handle imbalance.
    * Used `scikit-learn` **Pipelines** to combine preprocessing and modeling:
        * **Preprocessing:** `StandardScaler` for numerical features and `OneHotEncoder` for the categorical 'Type' feature.
        * **Models Tested:** Logistic Regression (baseline) and Random Forest Classifier.
    * Trained both baseline models.

3.  **Hyperparameter Tuning:** (`model.ipynb`)
    * Used `GridSearchCV` to optimize the hyperparameters of the Random Forest model, focusing on improving predictive performance (though initially optimized using 'accuracy', analysis focused on failure class metrics).

4.  **Evaluation:** (`model.ipynb`)
    * Compared models based on **Precision, Recall, and F1-Score** for the **'Failure' (Class 1)**, as accuracy is misleading due to imbalance.
    * Analyzed feature importances from the best-performing model (Tuned Random Forest).

## Results 📊

* **Logistic Regression** performed poorly, especially in identifying actual failures (Recall for Class 1: ~0.10, F1-Score: ~0.18).
* **Baseline Random Forest** significantly outperformed Logistic Regression (Recall: ~0.49, F1-Score: ~0.63 for Class 1).
* **Tuned Random Forest** (using GridSearchCV, optimized for accuracy but evaluated on F1/Recall) showed the best performance:
    * **Precision (Failure):** ~0.88
    * **Recall (Failure):** ~0.65
    * **F1-Score (Failure):** ~0.75
* **Key Features:** The most important features identified by the Random Forest model for predicting failure were **Torque**, **Rotational Speed**, and **Tool Wear**.

## Files in this Repository

* `eda.ipynb`: Jupyter Notebook containing the Exploratory Data Analysis.
* `model.ipynb`: Jupyter Notebook containing data preprocessing, model building, tuning, and evaluation.
* `data.csv`: The dataset used for the analysis (or provide link if hosted elsewhere).
* `requirements.txt`: List of Python libraries required to run the notebooks.
* `README.md`: This file explaining the project.
* *(Optional)* `rf_feature_importances.png`, `rf_confusion_matrix.png`, `log_reg_confusion_matrix.png`: Saved plots from the analysis.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/HrishiKShah28/Predictive-Maintenance-Sensor>
    cd <Predictive-Maintenance-Sensor>
    ```
2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch Jupyter Notebook or Jupyter Lab:**
    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```
5.  Open and run the cells in `eda.ipynb` and `model.ipynb`.

## Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
