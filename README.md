# Heart Disease Prediction — EDA & Feature Engineering

Exploratory data analysis, cleaning, feature engineering, and feature selection on the UCI-style heart disease dataset, in preparation for a binary classification model (presence vs. absence of heart disease).

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

1. **EDA** — shape, dtypes, summary stats, duplicate/null checks, distribution plots (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`), class balance of `HeartDisease`, categorical breakdowns (`Sex`, `ChestPainType`, `FastingBS`) against the target, box/violin plots, correlation heatmap.
2. **Cleaning** — `Cholesterol` and `RestingBP` contain invalid `0` values, imputed with the column mean (excluding the zeros).
3. **Preprocessing** — one-hot encoding of categorical columns.
4. **Feature engineering** — derived `AgeGroup` (Young / Middle-aged / Senior) and `CholesterolLevel` bins, one-hot encoded, then `StandardScaler` applied to numeric columns.
5. **Feature selection** — Pearson correlation for numeric features and Chi-square test of independence for categorical features against `HeartDisease`; features with no significant association (e.g. `ChestPainType_TA`, p = 0.13) are dropped.

## Status

EDA, cleaning, feature engineering, feature selection, and model training (KNN) are complete. A Streamlit web application (`app.py`) has been built for interactive predictions!

## Getting started

```bash
# Install the required dependencies
pip install -r requirements.txt

# Run the Streamlit web application
streamlit run app.py

# Or explore the notebook
jupyter notebook heart.ipynb
```

> Note: the notebook also installs an optional helper package, `sheryanalysis`, used for a quick automated EDA summary. It isn't required to reproduce the rest of the analysis.
