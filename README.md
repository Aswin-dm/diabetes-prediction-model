# Diabetes Prediction

This repository contains a Jupyter Notebook designed for predicting the occurrence of diabetes based on diagnostic measurements. The analysis and models are built using standard Python data science and machine learning libraries.

## Project Structure

- **[Exposys Dia.ipynb](file:///d:/aswin_vs/diabetes%20prediction/Exposys%20Dia.ipynb)**: The main Jupyter Notebook containing data exploration, preprocessing pipelines, feature engineering, and model evaluation.

## Dataset

The notebook is configured to work with the **Pima Indians Diabetes Dataset** (historically loaded from `../input/diabetes/diabetes.csv` in a Kaggle environment).

The dataset consists of **768 records** and **9 columns**:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function (genetic risk score)
- `Age`: Age (years)
- `Outcome`: Class variable (0 if non-diabetic, 1 if diabetic)

## Machine Learning Pipelines

The project evaluates three different machine learning pipelines using **10-fold Cross-Validation**:

### 1. Standardization + Linear Discriminant Analysis (LDA)
- **Preprocessing**: Standardizes features using `StandardScaler`.
- **Classifier**: `LinearDiscriminantAnalysis`.
- **Cross-Validation Accuracy**: **~77.35%**

### 2. Feature Union + Logistic Regression
- **Feature Extraction**: Combines components from Principal Component Analysis (`PCA` with `n_components=3`) and univariate feature selection (`SelectKBest` with `k=6`) using `FeatureUnion`.
- **Classifier**: `LogisticRegression`.
- **Cross-Validation Accuracy**: **~77.86%** (Best performing model)

### 3. Feature Union + XGBoost
- **Feature Extraction**: Same `FeatureUnion` setup (`PCA` + `SelectKBest`).
- **Classifier**: `XGBClassifier`.
- **Cross-Validation Accuracy**: **~75.00%**

## Getting Started

### Prerequisites

To run the notebook locally, you need Python installed along with the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `jupyter` / `notebook`

You can install the dependencies via pip:
```bash
pip install numpy pandas scikit-learn xgboost notebook
```

### Execution

1. Clone or download this repository.
2. Ensure you have the dataset `diabetes.csv`.
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Exposys Dia.ipynb"
   ```
4. Adjust the dataset path in the notebook if necessary (e.g., pointing `pd.read_csv` to your local `diabetes.csv` path instead of `../input/diabetes/diabetes.csv`).
5. Run the cells sequentially to reproduce the analysis.
