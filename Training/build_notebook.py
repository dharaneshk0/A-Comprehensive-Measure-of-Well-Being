"""Builds HumDevIndex.ipynb covering Epics 2-7 of the HDI project."""
import json
import uuid


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "id": uuid.uuid4().hex[:8],
        "source": source if isinstance(source, list) else source.split("\n"),
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": uuid.uuid4().hex[:8],
        "source": text if isinstance(text, list) else text.split("\n"),
    }


cells = []

# ---------------- TITLE ----------------
cells.append(md_cell(
    "# Human Development Index (HDI) Prediction\n"
    "### Machine Learning + Flask End-to-End Project\n\n"
    "This notebook walks through the full ML lifecycle to build a **Linear Regression** model "
    "that predicts a country's **HDI score** from four indicators:\n"
    "1. Life expectancy\n"
    "2. Mean years of schooling\n"
    "3. Expected years of schooling\n"
    "4. Gross National Income (GNI) per capita\n\n"
    "Workflow covered:\n"
    "- Epic 2: Import libraries\n"
    "- Epic 3: Dataset understanding & visualization\n"
    "- Epic 4: Data preprocessing & label encoding\n"
    "- Epic 5: Train / Test split\n"
    "- Epic 6: Fit Linear Regression & evaluate\n"
    "- Epic 7: Save the model with Pickle"
))

# ---------------- EPIC 2 ----------------
cells.append(md_cell("## Epic 2 — Importing Required Libraries"))
cells.append(code_cell(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.linear_model import LinearRegression\n"
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
    "from sklearn.preprocessing import LabelEncoder\n"
    "import pickle\n"
    "\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "# Display plots inline and set a clean style\n"
    "%matplotlib inline\n"
    "sns.set_style('whitegrid')\n"
    "print('All libraries imported successfully.')"
))

# ---------------- EPIC 3 ----------------
cells.append(md_cell(
    "## Epic 3 — Dataset Download and Understanding\n\n"
    "### Story 1 & 2: Load the dataset and explore its structure"
))
cells.append(code_cell(
    "# Load the HDI dataset\n"
    "df = pd.read_csv('../Dataset/HDI.csv')\n"
    "df.head()"
))
cells.append(code_cell(
    "# Shape of the dataset\n"
    "print('Rows    :', df.shape[0])\n"
    "print('Columns :', df.shape[1])"
))
cells.append(code_cell(
    "# Data types and non-null counts\n"
    "df.info()"
))
cells.append(code_cell(
    "# Statistical summary of numeric features\n"
    "df.describe()"
))

# ---------------- EPIC 3: Visualization ----------------
cells.append(md_cell("### Story 3: Data Visualization"))
cells.append(code_cell(
    "# Distribution of the target variable (HDI)\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.histplot(df['HDI'], bins=25, kde=True, color='steelblue')\n"
    "plt.title('Distribution of HDI Scores', fontsize=14)\n"
    "plt.xlabel('HDI')\n"
    "plt.ylabel('Frequency')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))
cells.append(code_cell(
    "# Pairwise relationships between indicators and HDI\n"
    "sns.pairplot(df.drop(columns=['Country']))\n"
    "plt.suptitle('Pairplot of HDI Indicators', y=1.02)\n"
    "plt.show()"
))
cells.append(code_cell(
    "# Correlation heatmap\n"
    "plt.figure(figsize=(7, 5))\n"
    "corr = df.drop(columns=['Country']).corr()\n"
    "sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f')\n"
    "plt.title('Feature Correlation Heatmap')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ---------------- EPIC 4 ----------------
cells.append(md_cell(
    "## Epic 4 — Data Preprocessing and Label Encoding\n\n"
    "### Story 1: Select dependent and independent variables\n\n"
    "- **Independent variables (X):** Life expectancy, Mean years of schooling, "
    "Expected years of schooling, GNI per capita\n"
    "- **Dependent variable (y):** HDI"
))
cells.append(code_cell(
    "# Independent features\n"
    "X = df[['Life expectancy',\n"
    "        'Mean years of schooling',\n"
    "        'Expected years of schooling',\n"
    "        'Gross National Income (GNI) per capita']]\n"
    "\n"
    "# Target variable\n"
    "y = df['HDI']\n"
    "\n"
    "X.head()"
))

cells.append(md_cell("### Story 2: Check for missing values"))
cells.append(code_cell(
    "# Check for nulls across the whole dataset\n"
    "df.isnull().sum()"
))

cells.append(md_cell(
    "### Story 3: Label Encoding\n\n"
    "The `Country` column is categorical text. We convert it to numerical labels so it can be "
    "used/saved alongside the model. The numeric indicators (already numeric) feed the regression directly."
))
cells.append(code_cell(
    "# Label-encode the Country column (categorical -> numeric)\n"
    "le = LabelEncoder()\n"
    "df['Country_encoded'] = le.fit_transform(df['Country'])\n"
    "\n"
    "# Preview encoded values\n"
    "df[['Country', 'Country_encoded']].head(10)"
))
cells.append(md_cell("### Story 4: Confirm the cleaned dataset is ready"))
cells.append(code_cell(
    "print('Final feature matrix shape :', X.shape)\n"
    "print('Target vector shape        :', y.shape)\n"
    "df_clean = X.copy()\n"
    "df_clean['HDI'] = y\n"
    "df_clean.head()"
))

# ---------------- EPIC 5 ----------------
cells.append(md_cell(
    "## Epic 5 — Train / Test Split\n\n"
    "Split the data so the model learns on one portion and is evaluated on unseen data."
))
cells.append(code_cell(
    "X_train, X_test, y_train, y_test = train_test_split(\n"
    "    X, y, test_size=0.2, random_state=42\n"
    ")\n"
    "\n"
    "print('Training samples :', X_train.shape[0])\n"
    "print('Testing samples  :', X_test.shape[0])"
))

# ---------------- EPIC 6 ----------------
cells.append(md_cell("## Epic 6 — Fitting the Model"))
cells.append(md_cell("### Story 1: Train the Linear Regression model"))
cells.append(code_cell(
    "model = LinearRegression()\n"
    "model.fit(X_train, y_train)\n"
    "print('Model trained successfully.')\n"
    "print('Coefficients :', model.coef_)\n"
    "print('Intercept    :', model.intercept_)"
))

cells.append(md_cell("### Story 2: Generate predictions"))
cells.append(code_cell(
    "y_pred = model.predict(X_test)\n"
    "\n"
    "# Compare actual vs predicted\n"
    "comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred.round(3)})\n"
    "comparison.head(10)"
))

cells.append(md_cell("### Story 3: Evaluate model performance"))
cells.append(code_cell(
    "mae = mean_absolute_error(y_test, y_pred)\n"
    "mse = mean_squared_error(y_test, y_pred)\n"
    "rmse = np.sqrt(mse)\n"
    "r2 = r2_score(y_test, y_pred)\n"
    "\n"
    "print(f'Mean Absolute Error (MAE) : {mae:.4f}')\n"
    "print(f'Mean Squared Error (MSE)  : {mse:.6f}')\n"
    "print(f'Root Mean Squared Error   : {rmse:.4f}')\n"
    "print(f'R-squared (R2) Score      : {r2:.4f}')"
))
cells.append(code_cell(
    "# Visualize actual vs predicted\n"
    "plt.figure(figsize=(8, 5))\n"
    "plt.scatter(y_test, y_pred, color='teal', alpha=0.7)\n"
    "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n"
    "plt.xlabel('Actual HDI')\n"
    "plt.ylabel('Predicted HDI')\n"
    "plt.title('Actual vs Predicted HDI')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

# ---------------- EPIC 7 ----------------
cells.append(md_cell("## Epic 7 — Saving the Model"))
cells.append(md_cell("### Story 1 & 2: Serialize with Pickle for reuse and deployment"))
cells.append(code_cell(
    "# Save the trained model so the Flask app can load it without retraining\n"
    "with open('../Flask/HDI.pkl', 'wb') as f:\n"
    "    pickle.dump(model, f)\n"
    "\n"
    "print('Model saved to ../Flask/HDI.pkl')\n"
    "\n"
    "# Quick sanity-check: reload and predict a sample\n"
    "with open('../Flask/HDI.pkl', 'rb') as f:\n"
    "    loaded_model = pickle.load(f)\n"
    "\n"
    "sample = pd.DataFrame([[82.6, 12.9, 18.2, 66000]],\n"
    "                      columns=X.columns)\n"
    "pred = loaded_model.predict(sample)[0]\n"
    "print(f'Sample prediction (Norway-like): HDI = {pred:.3f}')"
))

cells.append(md_cell(
    "---\n\n"
    "### ✅ Pipeline complete\n"
    "The trained Linear Regression model has been saved as `HDI.pkl`.\n"
    "The next step (Epic 8) is to build the Flask web application that uses this model "
    "to serve live HDI predictions to users."
))

# Assemble notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.2"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Normalize source lists to strings with newlines
for c in notebook["cells"]:
    src = c["source"]
    if isinstance(src, list):
        joined = "\n".join(src)
        # ensure each line but the last ends with \n
        lines = joined.split("\n")
        normalized = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        c["source"] = normalized

with open("HumDevIndex.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Built HumDevIndex.ipynb with", len(cells), "cells")
