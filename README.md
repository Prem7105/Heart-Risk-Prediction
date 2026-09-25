# Heart Risk Prediction Web Application

An end-to-end Machine Learning project to predict the risk of heart disease based on patient data. This project includes Exploratory Data Analysis (EDA), data cleaning, feature engineering, model selection, and a fully interactive **Streamlit web application** for making real-time predictions.

## Dataset

`heart.csv` — 918 patient records, 11 features + 1 target.

| Column | Description |
|---|---|
| Age | Age of the patient (years) |
| Sex | M / F |
| ChestPainType | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mm/dl) |
| FastingBS | Fasting blood sugar (1 if > 120 mg/dl, else 0) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y / N |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, Down |
| **HeartDisease** | Target: 1 = heart disease, 0 = normal |

## What's in the notebook (`heart.ipynb`)

1. **EDA** — shape, dtypes, summary stats, duplicate/null checks, distribution plots (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`), class balance of `HeartDisease`, categorical breakdowns (`Sex`, `ChestPainType`, `FastingBS`) against the target, box/violin plots, and a correlation heatmap.
2. **Cleaning** — `Cholesterol` and `RestingBP` contain invalid `0` values, which were imputed with the column mean (excluding the zeros).
3. **Preprocessing** — One-hot encoding of categorical columns.
4. **Feature Engineering** — Derived `AgeGroup` (Young / Middle-aged / Senior) and `CholesterolLevel` bins, which were one-hot encoded. A `StandardScaler` was then applied to numeric columns.
5. **Feature Selection** — Evaluated via Pearson correlation for numeric features and Chi-square test of independence for categorical features against `HeartDisease`; features with no significant association (e.g., `ChestPainType_TA`, p = 0.13) were dropped.

## Model Training & Evaluation

To find the most accurate algorithm for predictions, 5 different classification models were trained and evaluated on a 20% test split:

| Model | Accuracy | F1 Score |
|---|---|---|
| **KNN (K-Nearest Neighbors)** | **86.96%** | **0.8889** |
| Logistic Regression | 86.41% | 0.8804 |
| SVM (Support Vector Machine) | 85.87% | 0.8762 |
| Naive Bayes | 85.33% | 0.8670 |
| Decision Tree | 77.17% | 0.7941 |

**K-Nearest Neighbors (KNN)** performed the best and was selected as the final production model. The trained model (`KNN_heart.pkl`), column structure (`columns.pkl`), and data scaler (`scaler.pkl`) were exported to power the web application.

## Getting Started

Follow these steps to run the interactive Streamlit app on your local machine:

```bash
# Clone the repository
git clone https://github.com/Prem7105/Heart-Risk-Prediction.git

# Navigate to the project directory
cd Heart-Risk-Prediction

# Install the required dependencies
pip install -r requirements.txt

# Run the Streamlit web application
streamlit run app.py
```

> Note: The notebook also uses an optional helper package, `sheryanalysis`, for a quick automated EDA summary. It isn't required to run the main Streamlit app.
