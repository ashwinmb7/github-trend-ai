# GitHub Trending AI - Predicting Repository Growth

A machine learning project to predict how many stars a GitHub repository will gain based on its current metrics. This was my first ML project, focused on learning the end-to-end data science workflow.

## Project Goal

Predict `stars_period` (the number of stars a repository gains in a given period) using features like current stars, forks, contributors, programming language, and timeframe.

## Project Structure

```
github-trend-ai/
├── data/
│   ├── raw/                          # Original scraped data
│   └── processed/                    # Cleaned & engineered features
└── notebooks/
    ├── eda.ipynb                     # Exploratory Data Analysis
    ├── feature_engineering.ipynb     # Creating & normalizing features
    └── ml_model.ipynb                # Training ML models
```

## Methodology

### Phase 1: Exploratory Data Analysis
- Analyzed distributions and relationships in the data
- Identified missing values and outliers
- Found that the target variable (`stars_period`) is highly skewed

### Phase 2: Feature Engineering
- Created derived features: `stars_forks_ratio`, `log_stars`, `log_forks`, `popularity_score`
- Encoded categorical variables (language, timeframe)
- Normalized numerical features using StandardScaler
- Saved processed dataset for modeling

### Phase 3: Machine Learning Models
- Trained baseline models: Linear Regression, Random Forest, XGBoost
- Implemented improved models with log-transformed target: XGBoost, LightGBM, CatBoost
- Evaluated using MSE, RMSE, MAE, and R² scores

## Results

| Model | MSE | R² Score |
|-------|-----|----------|
| Linear Regression | ~1,283,743 | 0.01 |
| Random Forest | ~1,672,773 | -0.29 |
| XGBoost (Improved) | Varies | Low |

Model performance was poor across all approaches.

## Key Findings

Feature importance analysis revealed the model heavily depends on `forks`, `stars`, and their log/normalized versions. This indicates:

- The feature set lacks diversity
- Missing deeper repository dynamics: activity patterns, recency, growth trends, community engagement
- The model overfits to simple popularity features and fails to generalize

In essence, the model learns that "repos with more stars get more stars," which is true but not useful for prediction.

## Lessons Learned

**Technical Skills:**
- Data science workflow: EDA → Feature Engineering → Modeling → Evaluation
- Python libraries: pandas, scikit-learn, xgboost, lightgbm, catboost
- ML concepts: train/test split, feature scaling, target transformation, evaluation metrics

**Key Insights:**
- Feature quality and diversity matter more than data volume
- Domain knowledge is crucial for creating meaningful features
- Simple models should be tried before complex algorithms
- Feature engineering is often more important than algorithm choice
- Not every problem has a perfect ML solution

## Future Improvements

1. **Collect more diverse features:** repository age, commit frequency, issue/PR activity, recent updates
2. **Better feature engineering:** time-based features, growth rate features, interaction features
3. **Different approaches:** time series analysis, classification instead of regression, ensemble methods
4. **Better evaluation:** cross-validation, learning curves, residual analysis

## Technologies Used

- Python 3
- Jupyter Notebooks
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost

## Usage

1. Explore the data: Open `notebooks/eda.ipynb`
2. Feature engineering: Open `notebooks/feature_engineering.ipynb`
3. Train models: Open `notebooks/ml_model.ipynb`

## Note

This is a learning project. The models are not production-ready. The goal was to understand the ML workflow and learn from the challenges encountered.
