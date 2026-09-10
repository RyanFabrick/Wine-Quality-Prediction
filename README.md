# Wine Quality Prediction

This is a machine learning project predicting wine quality scores from objective physicochemical properties of red and white wines. The project uses a continuous regression framework across four models (Elastic Net, K-Nearest Neighbors, Random Forest, and Gradient Boosting) with 10-fold cross-validated hyperparameter tuning. It covers data cleaning, exploratory data analysis, preprocessing pipelines, model comparison, and feature importance analysis.

## Table of Contents

- [Why Did I Build This?](#why-did-i-build-this)
- [Results](#results)
- [Outline & Analysis](#outline--analysis)
- [Rendered Notebooks](#rendered-notebooks)
- [Dependencies](#dependencies)
- [License](#license)
- [Author](#author)
- [Acknowledgements & References](#acknowledgements--references)

## Why Did I Build This?

I built this as a final exam for an upper-division course (PSTAT 131: Introduction to Statistical Machine Learning) during my third year as a Statistics and Data Science major at UCSB. This project recieved an **A** letter grade. The inspiration came from a semester spent studying and living in Paris, France, where I explored French and greater European wine culture firsthand. I was fortunate enough to visit vineyards, wineries, and wine bars across Paris and Bordeaux. That experience got me interested in the science behind wine certification, which traditionally relies on subjective human expert tasters. This project asks whether quality scores can instead be predicted from objective, measurable chemistry alone.

## Results

| Model | Best Parameters | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|---|
| Elastic Net | alpha: 0.01, l1_ratio: 0.2 | 0.7281 | N/A | N/A |
| K-Nearest Neighbors | n_neighbors: 21, weights: distance | 0.6143 | 0.6168 | 0.5017 |
| Gradient Boosting | learning_rate: 0.1, max_depth: 7, n_estimators: 300 | 0.6182 | N/A | N/A |
| **Random Forest** | max_depth: 30, max_features: sqrt, n_estimators: 300 | **0.6028** | **0.6048** | **0.5209** |

*Only the top two cross-validated models (Random Forest, KNN) were evaluated on the held-out test set, per project requirements.*

**Key Finding**
- Random Forest was the best performer across every metric, explaining ~52% of the variance in wine quality scores (R² = 0.5209)
- Alcohol content was the single most important predictor (importance = 0.174), consistent with its strongest correlation to quality (r = 0.44) found during EDA
- Both non-linear models (Random Forest, KNN) meaningfully outperformed the linear Elastic Net baseline, confirming that wine chemistry–quality relationships are non-linear
- The model was most reliable for "average" wines (scores of 5–6) and regressed toward the mean at the extremes, a direct consequence of severe class imbalance in the target variable

## Outline & Analysis

For full code, statistical output, visualizations, and written interpretations, see the notebooks in [`Side Work/`](Side%20Work). For **rendered, readable versions,** see [Rendered Notebooks](#rendered-notebooks). The final, rendered version html file is [here](https://htmlpreview.github.io/?https://github.com/RyanFabrick/Wine-Quality-Prediction/blob/main/Rendered%20Notebooks/Final%20Written%20Report.html). A brief outline is below:

### Part 1: Introduction & Research Question
- Research question: can objective physicochemical properties predict inherently subject, human assigned wine quality scores?
- Dataset: combined red and white Vinho Verde wines (Cortez et al., 2009), 6,497 observations, 12 predictors, one target (quality, scored 0–10)
- Zero missing values across the entire dataset
- Regression framework chosen over classification to preserve variance across the full quality spectrum

### Part 2: Exploratory Data Analysis
- Quality scores are heavily concentrated at 5 and 6, with very few wines at the extremes (3, 4, 8, 9). This is consistent across both red and white types
- Correlation heatmap: alcohol most positively correlated with quality (r = 0.44); volatile acidity most negatively correlated (r = -0.27); strong collinearity between density and alcohol (r = -0.69)
- Several predictors (residual sugar, chlorides, free sulfur dioxide, sulphates) are heavily right skewed, motivating a log transformation
- Boxplots confirm alcohol's positive trend and volatile acidity's negative trend with quality hold across both wine types

### Part 3: Preprocessing & Cross-Validation Strategy
- Stratified 80/20 train test split (N=5,197 / N=1,300) to preserve the rare quality classes across both sets
- ColumnTransformer pipeline: log-transform + scale skewed features, standard-scale the rest
- 10-fold cross-validation (shuffled) used for all hyperparameter tuning to prevent data leakage and overfitting

### Part 4: Model Training & Tuning
- Four pipelines built and tuned via **GridSearchCV** (scored on negative MSE): Elastic Net, KNN, Random Forest, Gradient Boosting
- Random Forest achieved the lowest CV RMSE (0.6028), followed by KNN (0.6143), Gradient Boosting (0.6182), and Elastic Net (0.7281)
- Elastic Net's optimal l1_ratio (0.2) leans heavily on the Ridge penalty, consistent with the multicollinearity found during EDA

### Part 5: Test Set Evaluation & Feature Importance
- Random Forest and KNN (top two CV performers) evaluated on the held out test set; both generalized well with minimal CV-to-test RMSE gap
- Random Forest feature importances: alcohol (0.174) is the top predictor, followed by volatile acidity (0.111) and density (0.110); wine type (red vs. white) contributed almost nothing (0.005)
- Predicted-vs-actual and residual plots show strong performance in the 5–6 range, with systematic regression toward the mean at the quality extremes

### Part 6: Conclusions & Limitations
- Objective physicochemical properties can meaningfully predict wine quality (R² ≈ 0.52), though prediction at the extremes remains limited by data scarcity
- Severe class imbalance in quality scores (0.08% of wines score a 9) constrains how well the model can learn rare, high/low-quality profiles
- Wine type (red/white) adds negligible predictive power once chemistry is accounted for
- Future work: address class imbalance directly (e.g. resampling), test additional non-linear models, incorporate grape variety or vintage if available

## Rendered Notebooks

For full rendered HTML versions with all code, output, and visualizations, click below (opens via [htmlpreview.github.io](https://htmlpreview.github.io/) since GitHub doesn't render raw HTML files directly):
- [EDA Notebook](https://htmlpreview.github.io/?https://github.com/RyanFabrick/Wine-Quality-Prediction/blob/main/Rendered%20Notebooks/EDA_Notebook.html)
- [Preprocessing & Modeling Notebook](https://htmlpreview.github.io/?https://github.com/RyanFabrick/Wine-Quality-Prediction/blob/main/Rendered%20Notebooks/Preprocessing%26Modeling_Notebook.html)
- [Final Written Report](https://htmlpreview.github.io/?https://github.com/RyanFabrick/Wine-Quality-Prediction/blob/main/Rendered%20Notebooks/Final%20Written%20Report.html)
- [Codebook](https://htmlpreview.github.io/?https://github.com/RyanFabrick/Wine-Quality-Prediction/blob/main/Rendered%20Notebooks/codebook.html)

## Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
```

## License

**© 2026 Ryan Fabrick. All rights reserved. This project may not be reused, adapted, or submitted for academic coursework without explicit written permission from the author.**

## Author

**Ryan Fabrick**
- Statistics and Data Science (B.S) Student, University of California Santa Barbara
- GitHub: [https://github.com/RyanFabrick](https://github.com/RyanFabrick)
- LinkedIn: [www.linkedin.com/in/ryan-fabrick](https://www.linkedin.com/in/ryan-fabrick)
- Email: ryanfabrick@gmail.com

## Acknowledgements & References

- **[Cortez et al. (2009)](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub)** — *"Modeling wine preferences by data mining from physicochemical properties,"* the original source and collection of the Vinho Verde wine quality dataset
- **[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)** / **[Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)** — public hosting of the red and white wine datasets used in this project
________________________________________________
Built with ❤️ for UCSB

This project demonstrates my interest in machine learning, applied data science, and predictive modeling. It was completed for a final exam for an undergraduate upper division course in Statistical Machine Learning (PSTAT 131) where I recieved an A letter grade.