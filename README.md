# GitHub Trending AI - Predicting Repository Growth 🌟

> **My first AI/ML project!** A journey into machine learning to predict how many stars a GitHub repository will gain based on its current metrics.

## 📖 About This Project

This was my first dive into machine learning and AI. I wanted to understand if I could predict how popular a GitHub repository would become by looking at its current stats (stars, forks, contributors, etc.). 

**Spoiler alert:** It's harder than I thought! 😅 But I learned a ton along the way.

## 🎯 Project Goal

Predict `stars_period` (the number of stars a repository gains in a given period) using features like:
- Current stars and forks
- Number of contributors
- Programming language
- Timeframe (daily/weekly/monthly trending)

## 📁 Project Structure

```
github-trend-ai/
│
├── data/
│   ├── raw/                          # Original scraped data
│   │   ├── github_trending_repos.csv
│   │   ├── language_summary.csv
│   │   └── top_owners.csv
│   │
│   └── processed/                    # Cleaned & engineered features
│       ├── features.csv
│       ├── target.csv
│       └── processed_dataset.csv
│
└── notebooks/
    ├── eda.ipynb                     # 📊 Exploratory Data Analysis
    ├── feature_engineering.ipynb     # 🔧 Creating & normalizing features
    └── ml_model.ipynb                # 🤖 Training ML models
```

## 🚀 My Learning Journey

### Phase 1: Exploratory Data Analysis (EDA)
**Notebook:** `eda.ipynb`

**What I did:**
- Loaded and explored the raw GitHub trending repositories data
- Checked for missing values
- Analyzed distributions of stars, forks, and other metrics
- Visualized relationships between variables
- Grouped data by programming language

**What I learned:**
- Data is messy! Always check for missing values and outliers
- Visualizations help you understand your data before modeling
- Some languages are more popular than others (Python, JavaScript dominate)
- The target variable (`stars_period`) is highly skewed - most repos gain few stars, but some gain thousands

### Phase 2: Feature Engineering
**Notebook:** `feature_engineering.ipynb`

**What I did:**
- Selected relevant features (dropped names, URLs, etc.)
- Created new features:
  - `stars_forks_ratio` - Engagement metric
  - `log_stars`, `log_forks` - Log transforms for skewed data
  - `stars_per_contributor` - Efficiency metric
  - `popularity_score` - Combined popularity metric
  - Binary features (high_stars, high_forks, etc.)
- Encoded categorical variables (language, timeframe, search_language)
- Normalized all numerical features using StandardScaler
- Saved processed dataset for modeling

**What I learned:**
- Feature engineering is crucial! Raw data often needs transformation
- Log transforms help with skewed distributions
- Normalization (scaling) is important for ML models
- Creating new features from existing ones can improve predictions
- Categorical data needs encoding (I used Label Encoding)

### Phase 3: Machine Learning Models
**Notebook:** `ml_model.ipynb`

**What I did:**
- Split data into training (80%) and testing (20%) sets
- Trained baseline models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost
- Tried improved models with target transformation:
  - XGBoost with log-transformed target
  - LightGBM
  - CatBoost
- Evaluated using MSE, RMSE, MAE, and R² scores

**What I learned:**
- Train/test split prevents overfitting
- Baseline models help establish a starting point
- Target transformation (log) is important for skewed targets
- Different algorithms have different strengths
- Metrics matter: MSE, RMSE, MAE, R² each tell different stories

## 😅 The Reality Check

### What Worked Well ✅
- Data preprocessing pipeline
- Feature engineering (creating new features)
- Understanding the ML workflow
- Learning about different algorithms

### What Didn't Work So Well ❌
- **Poor model performance:** R² scores were very low (around 0.01 to negative!)
- **High MSE values:** Models struggled to predict accurately
- **Overfitting to simple features:** Models heavily relied on `forks`, `stars`, and their transformations

### Key Finding 🔍

**Feature importance analysis revealed:**
The model heavily depends on `forks`, `stars`, and their log/normalized versions. This indicates that:
- The current feature set lacks diversity
- We're missing deeper repository dynamics like:
  - Activity patterns (commits over time)
  - Recency (how recently was it updated?)
  - Growth trends (is it accelerating or slowing?)
  - Community engagement (issues, PRs, discussions)
  - Repository age and maturity
- The model overfits to these simple popularity features
- It fails to generalize, explaining the poor test performance

**In simple terms:** The model is basically saying "repos with more stars get more stars" - which is true but not very useful for prediction! 😅

## 🎓 What I Learned (The Big Picture)

### Technical Skills
1. **Data Science Workflow:**
   - EDA → Feature Engineering → Modeling → Evaluation
2. **Python Libraries:**
   - `pandas` for data manipulation
   - `scikit-learn` for ML models
   - `xgboost`, `lightgbm`, `catboost` for advanced models
   - `numpy` for numerical operations
3. **ML Concepts:**
   - Train/test split
   - Feature scaling/normalization
   - Target transformation
   - Model evaluation metrics
   - Overfitting vs underfitting

### Important Lessons
1. **More data ≠ better model** - Quality and diversity of features matter more
2. **Domain knowledge is crucial** - Understanding what makes repos popular helps create better features
3. **Simple models first** - Start with baselines before trying complex algorithms
4. **Feature engineering is an art** - Creating the right features is often more important than the algorithm
5. **Real-world ML is hard** - Not every problem can be solved with ML, and that's okay!

## 🔮 What I'd Do Differently Next Time

1. **Collect more diverse features:**
   - Repository age
   - Commit frequency
   - Issue/PR activity
   - Recent updates
   - Documentation quality
   - License type
   - Community size

2. **Better feature engineering:**
   - Time-based features (days since last update)
   - Growth rate features (stars per day)
   - Interaction features (stars × forks × contributors)

3. **Try different approaches:**
   - Time series analysis (if we had temporal data)
   - Classification instead of regression (predict categories like "high growth" vs "low growth")
   - Ensemble methods combining multiple models

4. **More thorough evaluation:**
   - Cross-validation
   - Learning curves
   - Residual analysis
   - Feature importance visualization

## 📊 Results Summary

| Model | MSE | R² Score | Status |
|-------|-----|----------|--------|
| Linear Regression | ~1,283,743 | 0.01 | ❌ Poor |
| Random Forest | ~1,672,773 | -0.29 | ❌ Very Poor |
| XGBoost (Original) | Varies | Low | ⚠️ Needs improvement |
| XGBoost (Improved) | Better | Better | ⚠️ Still needs work |

**Bottom line:** The models work, but they're not great at predicting. This is a learning project, and that's perfectly fine! 🎯

## 🛠️ Technologies Used

- **Python 3**
- **Jupyter Notebooks**
- **Libraries:**
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost

## 📝 How to Use This Project

1. **Explore the data:**
   ```bash
   # Open notebooks/eda.ipynb
   # Run cells to see data exploration
   ```

2. **Understand feature engineering:**
   ```bash
   # Open notebooks/feature_engineering.ipynb
   # See how features are created and normalized
   ```

3. **Train models:**
   ```bash
   # Open notebooks/ml_model.ipynb
   # Run cells to train and evaluate models
   ```

## 💭 Final Thoughts

This project taught me that machine learning is both exciting and humbling. Not every problem has a perfect ML solution, and that's part of the learning process. 

The most valuable lesson? **Understanding your data and domain is more important than fancy algorithms.** Sometimes the best model is the one that teaches you something about your problem, even if it doesn't have perfect accuracy.

I'm still learning, and this project is a snapshot of my journey. If you're also starting out, I hope this helps you understand that it's okay to struggle, experiment, and learn from "failed" models! 🚀

---

**Note:** This is a learning project. The models aren't production-ready, and that's intentional. The goal was to learn, not to build a perfect predictor. If you have suggestions or feedback, I'd love to hear them! 😊

