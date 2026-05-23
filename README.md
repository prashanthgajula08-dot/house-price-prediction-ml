# House Price Prediction (Linear Regression Model)

This repository contains a professional implementation of a Machine Learning model to predict house prices using the Ames Housing Dataset. It includes the original exploratory Jupyter notebook, modular training scripts, and output visual reports.

---

## 📂 Repository Structure

The project is organized as follows:
* **`data/`**: Contains the training dataset (`train.csv`).
* **`notebooks/`**: Contains the original interactive Jupyter notebook (`House price ML.ipynb`).
* **`src/`**: Contains clean, modular Python source scripts for pipeline automation:
  * `data_loader.py`: Handles CSV loading, missing values, and splitting.
  * `model.py`: Automates training, calculates evaluation metrics, and extracts feature importances.
  * `train.py`: Coordinates the entire training pipeline, prints logs, and outputs charts.
* **`reports/figures/`**: Generated plotting charts showing residuals and feature coefficients.
* **`requirements.txt`**: Standard list of required package dependencies.
* **`.gitignore`**: Ignores temporary Python cache and checkpoint files.

---

## 📊 Dataset & Feature Selection

The baseline model is trained using **5 key house features** from the dataset to predict the target label **`SalePrice`**:
1. **`OverallQual`**: Rates the overall material and finish of the house (1 to 10).
2. **`TotRmsAbvGrd`**: Total rooms above grade (does not include bathrooms).
3. **`YearBuilt`**: Original construction date.
4. **`GarageArea`**: Size of garage in square feet.
5. **`LotArea`**: Lot size in square feet.

---

## 📈 Model Performance & Evaluation

The model is a standard **Linear Regression** trained on 80% of the data and evaluated on the remaining 20% test subset.

### Evaluation Metrics:
* **Mean Absolute Error (MAE):** $27,463.66
* **Root Mean Squared Error (RMSE):** $44,837.32
* **R² Score:** 0.7379 (~73.8% predictive accuracy / variance explained)

### Feature Coefficients (Model Weight & Impact):
| Feature | Coefficient | Interpretation |
| :--- | :--- | :--- |
| **`OverallQual`** | +$28,683.40 | For every 1-point increase in Overall Quality, price increases by ~$28.6k |
| **`TotRmsAbvGrd`** | +$10,056.59 | For every extra room above grade, price increases by ~$10k |
| **`YearBuilt`** | +$324.27 | Houses increase in value by ~$324 for each year newer they are |
| **`GarageArea`** | +$69.14 | Every additional square foot of garage area increases price by ~$69 |
| **`LotArea`** | +$1.00 | Every additional square foot of lot area increases price by ~$1.00 |

*Intercept: Approximately -$624k (base value calculation factor).*

---

## 💻 Setup & Execution Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Clone this repository, navigate to the directory, and install requirements:
```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
Run the modular source script to load the data, train the model, evaluate metrics, and save output visualization plots:
```bash
python src/train.py
```

Upon successful run, the metrics will print to the console, and two visualization charts will be saved in `reports/figures/`:
* `coefficient_importance.png`: Visualizes feature coefficients in a bar chart.
* `residuals_distribution.png`: Displays error distributions (actual - predicted values).
