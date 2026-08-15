# Load necessary libraries
install.packages("readxl")
install.packages("openxlsx")
library(openxlsx)
library(readxl)
library(randomForest)
library(caret)
install.packages("gbm")
library(gbm)
library(ggplot2)
library(dplyr)

# Load the dataset
movie_data <- read.xlsx("Desktop/R assignment/project/simulated_movie_recommendation_dataset_600_records.xlsx")

# Exploratory Data Analysis (EDA)
str(movie_data)
summary(movie_data)

# View first few rows of the dataset
head(movie_data)

# Check unique genres
unique(movie_data$Genres)

# Visualize the distribution of movie ratings
ggplot(movie_data, aes(x = Rating)) +
  geom_histogram(binwidth = 0.5, fill = 'blue', color = 'black') +
  ggtitle('Distribution of Movie Ratings') +
  xlab('Rating') +
  ylab('Count')

# Visualize the number of movies per genre
genre_counts <- movie_data %>%
  group_by(Genres) %>%
  summarise(count = n())

ggplot(genre_counts, aes(x = Genres, y = count)) +
  geom_bar(stat = 'identity', fill = 'green', color = 'black') +
  coord_flip() +
  ggtitle('Number of Movies per Genre') +
  xlab('Genre') +
  ylab('Count')

# Check for missing values and handle them (imputation or removal)
sum(is.na(movie_data))

# Drop missing values for simplicity
movie_data <- na.omit(movie_data)

# Convert necessary columns to appropriate types
# Ensure that the User_ID column exists and convert it to a factor
movie_data$UserID <- as.factor(movie_data$UserID)

# Convert other necessary columns to appropriate types
movie_data$MovieID <- as.factor(movie_data$MovieID)
movie_data$ReleaseYear <- as.numeric(movie_data$ReleaseYear)
movie_data$Rating <- as.numeric(movie_data$Rating)

# Feature Engineering: Create new feature 'Release_Decade'
movie_data$Release_Decade <- floor(movie_data$ReleaseYear / 10) * 10

# Check correlations between numeric features
cor_matrix <- cor(movie_data %>% select_if(is.numeric))
print(cor_matrix)

# Visualize correlation heatmap
library(corrplot)
corrplot(cor_matrix, method = "circle")

# Split data into training and testing sets (80% train, 20% test)
set.seed(123)
trainIndex <- createDataPartition(movie_data$Rating, p = .8, list = FALSE, times = 1)
train_data <- movie_data[trainIndex, ]
test_data <- movie_data[-trainIndex, ]

# Preprocessing: Scale the numeric features
preProcValues <- preProcess(train_data[, "ReleaseYear", drop = FALSE], method = c("center", "scale"))
train_data_scaled <- predict(preProcValues, train_data)
test_data_scaled <- predict(preProcValues, test_data)

# Check summary of scaled data
summary(train_data_scaled)

# Model 1: Multilinear Regression
# Fit the linear model to predict ratings based on available features
lm_model <- lm(Rating ~ MovieID + ReleaseYear + Genres, data = train_data_scaled)
summary(lm_model)

# Predict on the test data using the linear model
predictions_lm <- predict(lm_model, test_data_scaled)

# Calculate performance metrics for Multilinear Regression
rmse_lm <- sqrt(mean((test_data_scaled$Rating - predictions_lm)^2))
mae_lm <- mean(abs(test_data_scaled$Rating - predictions_lm))
r2_lm <- cor(test_data_scaled$Rating, predictions_lm)^2

# Output the results of the model
cat("RMSE of Multilinear Regression:", rmse_lm, "\n")
cat("MAE of Multilinear Regression:", mae_lm, "\n")
cat("R² of Multilinear Regression:", r2_lm, "\n")

# Visualize the predictions vs actual ratings for Multilinear Regression
ggplot(test_data_scaled, aes(x = Rating, y = predictions_lm)) +
  geom_point() +
  geom_smooth(method = 'lm', formula = y ~ x) +
  ggtitle("Multilinear Regression: Actual vs Predicted Ratings") +
  xlab("Actual Rating") +
  ylab("Predicted Rating")

# Model 2: Random Forest with Cross-Validation
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)

# Create a tuning grid for hyperparameter search
tuneGrid_rf <- expand.grid(.mtry = c(2, 3, 4))

# Train Random Forest model with tuning
rf_model <- train(Rating ~ MovieID + ReleaseYear + Genres, data = train_data_scaled,
                  method = "rf", trControl = train_control, tuneGrid = tuneGrid_rf)

# Output best model from tuning
rf_model$bestTune
rf_model

# Predict using the Random Forest model
predictions_rf <- predict(rf_model, test_data_scaled)

# Calculate performance metrics for Random Forest
rmse_rf <- sqrt(mean((test_data_scaled$Rating - predictions_rf)^2))
mae_rf <- mean(abs(test_data_scaled$Rating - predictions_rf))
r2_rf <- cor(test_data_scaled$Rating, predictions_rf)^2

# Output Random Forest results
cat("RMSE of Random Forest:", rmse_rf, "\n")
cat("MAE of Random Forest:", mae_rf, "\n")
cat("R² of Random Forest:", r2_rf, "\n")


rf_model <- randomForest(Rating ~ Genres + ReleaseYear, data = train_data)
print(rf_model)

# Predictions for Random Forest
predictions_rf <- predict(rf_model, newdata = test_data)

# Check predictions
print(head(predictions_rf))

# Create a DataFrame for plotting
results_rf <- data.frame(Actual = test_data$Rating, Predicted = predictions_rf)

# Plot for Random Forest
rf_plot <- ggplot(results_rf, aes(x = Actual, y = Predicted)) +
  geom_point(color = "red") +
  geom_smooth(method = 'lm', color = 'darkred', se = FALSE) +
  labs(title = "Random Forest: Actual vs Predicted Ratings", 
       x = "Actual Ratings", 
       y = "Predicted Ratings")

# Print and save the plot
print(rf_plot)
ggsave("random_forest_actual_vs_predicted.png", plot = rf_plot, width = 8, height = 6)

# Model 3: Boosted Trees (GBM) with Tuning
set.seed(123)

# Create tuning grid for GBM
gbm_grid <- expand.grid(.n.trees = c(50, 100, 150),
                        .interaction.depth = c(1, 3, 5),
                        .shrinkage = c(0.01, 0.1),
                        .n.minobsinnode = 10)

# Train Boosted Tree model using the grid search
gbm_model <- train(Rating ~ MovieID + ReleaseYear + Genres, data = train_data_scaled,
                   method = "gbm", trControl = train_control, tuneGrid = gbm_grid, verbose = FALSE)

# Output best model from tuning
gbm_model$bestTune
gbm_model

# Predict using Boosted Trees model
predictions_gbm <- predict(gbm_model, test_data_scaled)

# Calculate performance metrics for Boosted Trees
rmse_gbm <- sqrt(mean((test_data_scaled$Rating - predictions_gbm)^2))
mae_gbm <- mean(abs(test_data_scaled$Rating - predictions_gbm))
r2_gbm <- cor(test_data_scaled$Rating, predictions_gbm)^2

# Output Boosted Trees results
cat("RMSE of Boosted Trees:", rmse_gbm, "\n")
cat("MAE of Boosted Trees:", mae_gbm, "\n")
cat("R² of Boosted Trees:", r2_gbm, "\n")


# Predictions for Boosted Trees
predictions_boosted <- predict(gbm_model, newdata = test_data_scaled)

# Check predictions
print(head(predictions_boosted))

# Create a DataFrame for plotting
results_boosted <- data.frame(Actual = test_data$Rating, Predicted = predictions_boosted)

# Plot for Boosted Trees
boosted_plot <- ggplot(results_boosted, aes(x = Actual, y = Predicted)) +
  geom_point(color = "green") +
  geom_smooth(method = 'lm', color = 'darkgreen', se = FALSE) +
  labs(title = "Boosted Trees: Actual vs Predicted Ratings", 
       x = "Actual Ratings", 
       y = "Predicted Ratings")

# Print and save the plot
print(boosted_plot)
ggsave("boosted_trees_actual_vs_predicted.png", plot = boosted_plot, width = 8, height = 6)

# Compare RMSE of all models
cat("RMSE of Multilinear Regression:", rmse_lm, "\n")
cat("RMSE of Random Forest:", rmse_rf, "\n")
cat("RMSE of Boosted Trees:", rmse_gbm, "\n")

# Compare MAE of all models
cat("MAE of Multilinear Regression:", mae_lm, "\n")
cat("MAE of Random Forest:", mae_rf, "\n")
cat("MAE of Boosted Trees:", mae_gbm, "\n")

# Compare R² of all models
cat("R² of Multilinear Regression:", r2_lm, "\n")
cat("R² of Random Forest:", r2_rf, "\n")
cat("R² of Boosted Trees:", r2_gbm, "\n")

