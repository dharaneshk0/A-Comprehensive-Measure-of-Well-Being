# 🌍 Human Development Index (HDI) Prediction

An end-to-end Machine Learning project that predicts a country's **Human Development Index (HDI)** from four socio-economic indicators using **Linear Regression**, served through a **Flask web application** with a premium 3D-animated UI.

![HDI Prediction App](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Workflow (End-to-End Pipeline)](#-workflow-end-to-end-pipeline)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Using the Web App](#-using-the-web-app)
- [Training Details](#-training-details)
- [Model Performance](#-model-performance)
- [UI/UX Features](#-uiux-features)
- [Testing](#-testing)
- [Customization](#-customization)
- [License](#-license)

---

## 🎯 Overview

The **Human Development Index (HDI)** is a summary measure of average achievement in key dimensions of human development, published by the United Nations Development Programme (UNDP). This project:

1. **Generates a realistic HDI dataset** based on the UNDP formula using real-world country data
2. **Trains a Linear Regression model** to predict HDI from four indicators
3. **Serves predictions** through a beautiful interactive web app with 3D visual effects

### The Four Indicators

| Indicator | Description | Typical Range |
|-----------|-------------|---------------|
| 📅 **Life Expectancy** | Average years a newborn is expected to live | 20–90 years |
| 📚 **Mean Years of Schooling** | Average years of education adults have received | 0–18 years |
| 🎓 **Expected Years of Schooling** | Years of education a child is expected to receive | 0–20 years |
| 💰 **GNI per Capita** | Gross National Income per person (US$) | $100–$100,000 |

### HDI Classification Tiers

| Score Range | Classification |
|-------------|----------------|
| 0.800 – 1.000 | 🟢 **Very High Human Development** |
| 0.700 – 0.799 | 🔵 **High Human Development** |
| 0.550 – 0.699 | 🟠 **Medium Human Development** |
| 0.000 – 0.549 | 🔴 **Low Human Development** |

---

## 📁 Project Structure

```
SmartBridge/
├── Dataset/
│   ├── generate_dataset.py    # Generates HDI.csv using UNDP formula
│   └── HDI.csv                # Generated dataset (124 countries)
│
├── Training/
│   ├── train_model.py          # End-to-end ML pipeline (load → train → save)
│   ├── build_notebook.py       # Programmatic Jupyter notebook builder
│   └── HumDevIndex.ipynb       # Generated notebook (Epics 2-7)
│
├── Flask/
│   ├── app.py                  # Flask web application
│   ├── HDI.pkl                 # Saved trained model (pickle)
│   ├── test_app.py             # Integration tests for Flask routes
│   ├── static/
│   │   ├── style.css           # Premium UI with 3D animations & glassmorphism
│   │   └── script.js           # Interactive JS (Three.js, card tilt, particles)
│   └── templates/
│       ├── index.html          # Input form with 3D particle background
│       └── result.html         # Result page with animated score counter
│
└── README.md                   # This file
```

---

## 🔄 Workflow (End-to-End Pipeline)

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Dataset Generation                                    │
│     └── generate_dataset.py                              │
│         └── Computes HDI using UNDP geometric mean       │
│             formula from real country data               │
│             → HDI.csv (124 rows × 6 columns)             │
│                                                          │
│  2. Model Training                                        │
│     └── train_model.py                                   │
│         ├── Loads HDI.csv                                │
│         ├── Selects features (4 indicators) & target     │
│         ├── Splits data (80% train / 20% test)           │
│         ├── Trains Linear Regression model               │
│         ├── Evaluates performance (MAE, RMSE, R²)        │
│         └── Serializes model → HDI.pkl                   │
│                                                          │
│  3. Notebook (Optional)                                   │
│     └── build_notebook.py                                │
│         └── Generates HumDevIndex.ipynb with             │
│             visualizations & detailed walkthrough        │
│                                                          │
│  4. Web Application                                       │
│     └── Flask/app.py                                     │
│         ├── Loads trained model from HDI.pkl             │
│         ├── Serves input form at GET /                   │
│         └── Returns prediction at POST /predict          │
│                                                          │
│  5. Testing                                               │
│     └── test_app.py                                      │
│         └── Validates all 3 HDI tiers with live server   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### HDI Formula (UNDP)

The dataset uses the official UNDP geometric mean formula:

```
Health Index       = (LifeExpectancy - 20) / (85 - 20)
Education Index    = (MeanSchooling / 15 + ExpectedSchooling / 18) / 2
Income Index       = (ln(GNI) - ln(100)) / (ln(75000) - ln(100))

HDI = (Health × Education × Income)^(1/3)
```

---

## ✅ Prerequisites

- **Python 3.10+** (tested on 3.13)
- **pip** (Python package manager)
- **Modern web browser** (Chrome, Firefox, Edge, Safari)

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SmartBridge
```

### 2. Install Dependencies

```bash
pip install flask numpy pandas scikit-learn requests matplotlib seaborn
```

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | ≥3.0 | Web framework |
| `numpy` | ≥2.0 | Numerical computations |
| `pandas` | ≥2.0 | Data manipulation |
| `scikit-learn` | ≥1.6 | Machine Learning (Linear Regression) |
| `requests` | ≥2.32 | HTTP client for testing |
| `matplotlib` | ≥3.10 | Plotting (notebook) |
| `seaborn` | ≥0.13 | Statistical visualizations (notebook) |

---

## 🚀 Running the Project

### Option 1: Run Everything (Full Pipeline)

```bash
# Step 1: Generate the dataset
cd Dataset
python generate_dataset.py

# Step 2: Train the model
cd ../Training
python train_model.py

# Step 3: (Optional) Build the Jupyter notebook
python build_notebook.py

# Step 4: Start the Flask app
cd ../Flask
python app.py
```

### Option 2: Quick Start (App Only)

If the dataset and model are already generated (`HDI.csv` and `HDI.pkl` exist):

```bash
cd Flask
python app.py
```

Then open your browser to: **http://127.0.0.1:5000**

### Option 3: Using the Jupyter Notebook

```bash
cd Training
python build_notebook.py
jupyter notebook HumDevIndex.ipynb
```

---

## 🌐 Using the Web App

### Input Page (`/`)

1. Enter the four indicators in the form:
   - **Life Expectancy** (e.g., 82.6 for Norway-like)
   - **Mean Years of Schooling** (e.g., 12.9)
   - **Expected Years of Schooling** (e.g., 18.2)
   - **GNI per Capita** (e.g., 66000)
2. Click **"Predict HDI"**
3. Watch the animated 3D background and card tilt effects as you interact

### Result Page (`/predict`)

The result page displays:
- **Animated HDI Score** — Counts up smoothly from 0 to your prediction
- **Development Tier Badge** — Color-coded with glow animation
- **HDI Gauge Bar** — Visual scale from Low to Very High
- **Input Summary** — Staggered entrance animation of your submitted values

### Example Predictions

| Scenario | Life Exp. | Mean School | Exp. School | GNI | Predicted HDI | Tier |
|----------|-----------|-------------|-------------|-----|---------------|------|
| 🇳🇴 Norway-like | 82.6 | 12.9 | 18.2 | $66,000 | **0.972** | 🟢 Very High |
| 🇧🇷 Emerging Economy | 72.5 | 8.0 | 13.0 | $11,000 | **0.696** | 🟠 Medium |
| 🇳🇪 Low Development | 60.0 | 3.5 | 8.0 | $2,000 | **0.457** | 🔴 Low |

---

## 🧠 Training Details

### Model

- **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Training/Test Split:** 80% / 20% (random_state=42)
- **Features:** 4 numerical indicators (no encoding needed)
- **Target:** HDI score (continuous, 0–1)

### Dataset

- **Size:** 124 countries
- **Source:** Realistic synthetic data based on real-world UNDP reported values
- **Range:** HDI from 0.365 (Somalia) to 0.986 (Australia)

---

## 📊 Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | **0.9889** | Model explains 98.89% of variance — excellent fit |
| **MAE** | **0.0136** | Average prediction error is ~0.014 HDI points |
| **RMSE** | **0.0161** | Consistent error distribution, no large outliers |

The model achieves near-perfect accuracy because the HDI formula is a deterministic function of the four indicators, and Linear Regression captures the multiplicative geometric mean relationship extremely well.

---

## 🎨 UI/UX Features

The web application features a premium visual design:

| Feature | Description |
|---------|-------------|
| 🌌 **3D Particle Background** | Three.js-powered particle system with 2000 colored particles responding to mouse movement |
| ✨ **Glassmorphism Cards** | Frosted glass effect with backdrop-blur, shimmer animations, and glow borders |
| 🎯 **Card 3D Tilt** | Cards dynamically rotate following the cursor with real-time shadow tracking |
| 🔘 **Interactive Buttons** | Gradient with shine sweep, ripple click effect, 3D hover lift |
| 📊 **Animated Score Counter** | Number smoothly animates from 0 to the predicted HDI |
| 🌈 **Tier Badge Glow** | Color-coded badges (green/blue/orange/red) with pulsing glow animations |
| 📈 **HDI Gauge Bar** | Visual progress bar showing score position on the development spectrum |
| 🎬 **Staggered Entrances** | Elements animate in sequentially using IntersectionObserver |
| 🎭 **Floating Orbs** | Ambient gradient orbs floating in the background |
| 📱 **Fully Responsive** | Adapts to all screen sizes with touch-device support |

### Tech Stack for UI

- **CSS3** — Custom properties, keyframe animations, gradients, glassmorphism
- **Three.js** (r128) — 3D particle system via CDN
- **Vanilla JavaScript** — IntersectionObserver, event handling, animation loops
- **No frontend framework** — Pure server-rendered Flask templates

---

## 🧪 Testing

### Integration Tests

Run the test suite against a running Flask server:

```bash
# Terminal 1: Start Flask
cd Flask
python app.py

# Terminal 2: Run tests
python test_app.py
```

The tests verify:
1. ✅ Home page renders at `GET /` (status 200, correct title)
2. ✅ Very High scenario → HDI ≥ 0.85, "Very High" tier
3. ✅ Medium scenario → HDI 0.55–0.70, "Medium" tier
4. ✅ Low scenario → HDI ≤ 0.55, "Low" tier

### Manual Testing with curl

```bash
# Check home page
curl http://127.0.0.1:5000/

# Test prediction
curl -X POST http://127.0.0.1:5000/predict \
  -d "life_expectancy=82.6&mean_schooling=12.9&expected_schooling=18.2&gni=66000"
```

---

## 🛠 Customization

### Adding Countries to the Dataset

Edit `Dataset/generate_dataset.py` and add new entries to the `data` list:

```python
data = [
    # (Country, LifeExpectancy, MeanSchooling, ExpectedSchooling, GNIperCapita)
    ("New Country", 75.0, 10.0, 14.0, 20000),
    # ... add more
]
```

Then regenerate the dataset and retrain:

```bash
cd Dataset && python generate_dataset.py
cd ../Training && python train_model.py
```

### Adding More Features

To add additional indicators (e.g., literacy rate, urbanization):

1. Add the column to the `data` list in `generate_dataset.py`
2. Add the column to the `feature_cols` list in `train_model.py`
3. Add a form field in `Flask/templates/index.html`
4. Update the `predict()` route in `Flask/app.py`

### Changing the Model

To try a different model (e.g., Random Forest):

1. Import the model in `train_model.py`
2. Replace `LinearRegression()` with the new model
3. Retrain and the Flask app will automatically pick up the new `HDI.pkl`

---

## 📄 License

This project is for educational and demonstration purposes. The dataset uses approximated real-world values for illustrative purposes.

---

<p align="center">
  Made with ❤️ for Human Development
</p>
