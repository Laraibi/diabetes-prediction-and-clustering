# 🩺 Diabetes Prediction and Clustering

An end-to-end Machine Learning project combining **unsupervised learning (K-Means clustering)** and **supervised learning (classification)** to analyze diabetes risk profiles from patient medical data.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Clustering with K-Means
- Cluster interpretation
- Multi-model classification
- Hyperparameter tuning
- Model export
- Interactive Streamlit deployment

---

# 📌 Project Objective

The goal of this project is to:

- identify hidden patient profiles using clustering
- classify patients into discovered risk groups
- evaluate multiple ML algorithms
- deploy an interactive prediction application

Unlike traditional diabetes prediction systems based only on labels, this project first discovers hidden structures inside the dataset using unsupervised learning.

---

# 📂 Dataset Features

The dataset contains medical indicators such as:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Glucose concentration |
| BloodPressure | Blood pressure |
| SkinThickness | Skin fold thickness |
| Insulin | Insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic diabetes likelihood |
| Age | Patient age |

---

# 🧠 Machine Learning Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
EDA & Visualization
   ↓
Missing Value Imputation
   ↓
Feature Scaling
   ↓
KMeans Clustering
   ↓
Cluster Analysis
   ↓
Cluster Classification
   ↓
Hyperparameter Tuning
   ↓
Model Export
   ↓
Streamlit Deployment

---

# 📊 Exploratory Data Analysis (EDA)

Performed analyses include:

- dataset overview
- missing value detection
- duplicate detection
- statistical summaries
- distribution histograms
- correlation heatmaps
- pairplots
- outlier detection using IQR and boxplots

---

# ⚙️ Data Preprocessing

## Missing Values

Some medical columns contained invalid zero values:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

These values were replaced with `NaN` and handled using:

```python
SimpleImputer(strategy="median")
```

---

## Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

This step is important for:
- KMeans clustering
- SVM
- Logistic Regression

---

# 🔍 Clustering with KMeans

## Elbow Method

The optimal number of clusters was determined using:
- inertia analysis
- elbow curve

---

## Silhouette Score

Silhouette analysis was used to validate cluster quality.

Final chosen value:

```text
k = 4
```

---

# 📈 Cluster Interpretation

Clusters were analyzed using feature averages.

The high-risk diabetes cluster was identified based on:

| Medical Indicator | Threshold |
|---|---|
| Glucose | > 126 |
| BMI | > 30 |
| DiabetesPedigreeFunction | > 0.5 |

This project combines:
- statistical clustering
- medical interpretation

to provide more realistic results.

---

# 🤖 Supervised Classification

After clustering:
- cluster labels became target labels
- supervised models learned to predict cluster membership

---

# 🧪 Tested Models

| Model | Purpose |
|---|---|
| Logistic Regression | Linear classification |
| Decision Tree | Tree-based classification |
| Random Forest | Ensemble learning |
| Gradient Boosting | Boosting algorithm |
| SVM | Margin-based classification |
| XGBoost | Advanced boosting |

---

# 📏 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# 🚀 Hyperparameter Tuning

Hyperparameter optimization was performed using:

```python
GridSearchCV
```

Optimized models include:
- Logistic Regression
- XGBoost

---

# 💾 Model Export

The following artifacts were exported using `joblib`:

| File | Description |
|---|---|
| diabetes_imputer.joblib | Missing value imputer |
| diabetes_scaler.joblib | Feature scaler |
| diabetes_kmeans.joblib | Clustering model |
| diabetes_classifier.joblib | Final classifier |
| high_risk_cluster.joblib | High-risk cluster identifier |

---

# 🌐 Streamlit Application

An interactive Streamlit application was developed to:

- enter patient medical information
- predict diabetes risk profile
- visualize results in real time

---

# 🩺 Medical Interpretation Layer

A hybrid medical interpretation layer was added.

This combines:
- machine learning predictions
- explicit medical rules

Example:

```python
if glucose > 126 and bmi > 30:
    risk = "Moderate Risk"
```

This improves:
- realism
- explainability
- healthcare relevance

---

# 🛠️ Technologies Used

| Technology | Role |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| XGBoost | Boosting algorithms |
| Imbalanced-learn | Oversampling |
| Joblib | Model serialization |
| Streamlit | Deployment UI |
| UV | Python environment management |
| Jupyter Notebook | Experimentation |

---

# 📁 Project Structure

```text
diabetes-prediction-and-clustering/
│
├── app/
│   └── app.py
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── diabetes_classifier.joblib
│   ├── diabetes_imputer.joblib
│   ├── diabetes_kmeans.joblib
│   ├── diabetes_scaler.joblib
│   └── high_risk_cluster.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_classification_models.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_model_export.ipynb
│
├── src/
├── reports/
├── README.md
├── pyproject.toml
└── uv.lock
```

---

# ▶️ Installation

## Clone repository

```bash
git clone https://github.com/Laraibi/diabetes-prediction-and-clustering.git
```

---

## Enter project

```bash
cd diabetes-prediction-and-clustering
```

---

## Install dependencies

```bash
uv sync
```

---

# ▶️ Run Streamlit App

```bash
uv run streamlit run app/app.py
```

---

# 📸 Application Preview

Add screenshots here:

```text
screenshots/app-preview.png
```

---

# 🔮 Future Improvements

Possible improvements:
- real medical labels integration
- advanced feature engineering
- SHAP explainability
- Docker deployment
- cloud deployment
- FastAPI backend
- deep learning experimentation
- authentication system
- patient history tracking

---

# 📚 Educational Goals

This project was developed for learning purposes to practice:

- Machine Learning workflows
- Data preprocessing
- Clustering
- Classification
- Hyperparameter tuning
- Model deployment
- Streamlit applications
- AI engineering best practices

---

# 👨‍💻 Author

## Mehdi Laraibi

Full Stack Developer & AI Enthusiast

- GitHub: https://github.com/Laraibi
- LinkedIn: https://www.linkedin.com/in/mehdi-laraibi-33703ab6

---

# ⭐ Final Result

This project demonstrates a complete Machine Learning pipeline from raw medical data to deployable AI application using both unsupervised and supervised learning approaches.