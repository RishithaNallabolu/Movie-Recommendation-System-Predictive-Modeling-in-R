# Movie Recommendation System — Predictive Modeling in R

A movie rating prediction project built in R, comparing three modeling approaches — Multilinear Regression, Random Forest, and Boosted Trees (GBM) — to predict how a user would rate a movie based on genre, release year, and other features.

## Dataset

A simulated dataset of 600 user movie-rating records, with the following key fields:

| Column | Description |
|--------|-------------|
| UserID | Unique identifier for each user |
| MovieID | Unique identifier for each movie |
| Genres | Movie genre(s) |
| ReleaseYear | Year the movie was released |
| Rating | User rating (scale of 1–5) — the target variable |

## Project workflow

**1. Exploratory Data Analysis**
Examined the structure and summary statistics of the dataset, checked unique genre categories, and visualized the distribution of ratings and the number of movies per genre.

**2. Data Cleaning**
Checked for and removed missing values. Outliers were filtered out to avoid distorting the analysis, and zero values were evaluated for validity within context (replaced with the mean rating where they represented missing data rather than genuine zero ratings).

**3. Feature Engineering**
Converted categorical fields (UserID, MovieID) to factors, and engineered a new `Release_Decade` feature from release year to capture broader era-based patterns.

**4. Correlation Analysis**
Examined correlations between numeric features and visualized them with a correlation heatmap.

**5. Modeling**
Split the data 80/20 into training and test sets, then trained and evaluated three models:

- **Multilinear Regression** — a simple, interpretable baseline to evaluate the relationship between rating and features like genre and release year.
- **Random Forest** — tuned via 5-fold cross-validation, chosen for its ability to capture non-linear relationships that linear regression misses.
- **Boosted Trees (GBM)** — tuned across tree count, interaction depth, and shrinkage, combining many weak learners to capture complex patterns in the data.

Each model was evaluated using RMSE, MAE, and R² on the held-out test set.

## Results

**Best performing model: Boosted Trees**

Boosted Trees produced the lowest RMSE of the three models, meaning its rating predictions were consistently closest to the actual ratings. It outperformed the simpler models by better capturing non-linear relationships between users, genres, and movies, and by generalizing better to unseen data through its iterative error-correction approach. Multilinear Regression had the weakest performance, limited by the fundamentally non-linear relationships in the data that a linear model can't capture.

## Real-world application

This modeling approach mirrors the foundation of recommendation systems used by streaming platforms — predicting how a user would rate content they haven't seen yet, based on patterns in genre, release timing, and user behavior. Natural next steps include collaborative filtering, incorporating user demographic features, and hyperparameter tuning for further accuracy gains.

## Tools

R, ggplot2, dplyr, randomForest, caret, gbm, corrplot
