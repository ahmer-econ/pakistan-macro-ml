install.packages("glmnet")
# ============================================================
# Project: Machine Learning Meets Econometrics
# Script 04: Ridge Regression
# Author: Ahmer | GitHub: ahmer-econ | August 2026
# ============================================================

setwd("D:/Documents/APPLIED ECONOMETRICS WORK/ML Model")

library(tidyverse)
library(glmnet)

# ============================================================
# SECTION 1: LOAD DATA
# ============================================================

train <- read.csv("data_clean/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("data_clean/test.csv",  stringsAsFactors = FALSE)
train$date <- as.Date(train$date)
test$date  <- as.Date(test$date)

predictors <- c("cpi_lag1","cpi_lag2","cpi_lag3",
                "m2_lag1","m2_lag2",
                "pkr_lag1","pkr_lag2",
                "rate_lag1","rate_lag2",
                "oil_lag1","oil_lag2",
                "fao_lag1","fao_lag2")

X_train <- as.matrix(train[, predictors])
y_train <- train$cpi
X_test  <- as.matrix(test[, predictors])
y_test  <- test$cpi

# ============================================================
# SECTION 2: RIDGE — CV LAMBDA SELECTION
# ============================================================

set.seed(42)
ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0,
                      nfolds = 10, standardize = TRUE)

best_lambda_ridge <- ridge_cv$lambda.min
cat("Best lambda (Ridge):", round(best_lambda_ridge, 4), "\n")

# ============================================================
# SECTION 3: FIT FINAL RIDGE MODEL
# ============================================================

ridge_model <- glmnet(X_train, y_train, alpha = 0,
                      lambda = best_lambda_ridge, standardize = TRUE)

# Coefficients
ridge_coef <- coef(ridge_model)
print(ridge_coef)

# ============================================================
# SECTION 4: OOS PREDICTION & METRICS
# ============================================================

test$ridge_pred <- as.numeric(predict(ridge_model, newx = X_test))

ridge_rmse <- sqrt(mean((y_test - test$ridge_pred)^2))
ridge_mae  <- mean(abs(y_test - test$ridge_pred))

cat("Ridge OOS RMSE:", round(ridge_rmse, 4), "\n")
cat("Ridge OOS MAE: ", round(ridge_mae,  4), "\n")

# ============================================================
# SECTION 5: PLOTS
# ============================================================

# Plot 1: CV lambda path
png("outputs/plots/05_ridge_cv_lambda.png", width = 800, height = 500, res = 120)
plot(ridge_cv)
title("Ridge — 10-Fold CV: MSE vs Log Lambda", line = 2.5)
dev.off()

# Plot 2: Coefficient path
ridge_path <- glmnet(X_train, y_train, alpha = 0, standardize = TRUE)
png("outputs/plots/06_ridge_coef_path.png", width = 800, height = 500, res = 120)
plot(ridge_path, xvar = "lambda", label = TRUE)
title("Ridge — Coefficient Paths", line = 2.5)
dev.off()

# Plot 3: Actual vs Predicted
ggplot() +
  geom_line(data = train, aes(x = date, y = cpi), color = "grey60", linewidth = 0.6) +
  geom_line(data = test, aes(x = date, y = cpi),        color = "#1565C0", linewidth = 1) +
  geom_line(data = test, aes(x = date, y = ridge_pred), color = "#E65100", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dotted") +
  labs(title = "Ridge Regression — Actual vs Predicted CPI (2023 Holdout)",
       subtitle = "Blue = Actual | Orange dashed = Ridge Predicted | Grey = Training",
       x = NULL, y = "CPI YoY %") +
  theme_minimal(base_size = 11)

ggsave("outputs/plots/07_ridge_predictions.png", width = 10, height = 5, dpi = 150)
cat("Script 04 complete.\n")