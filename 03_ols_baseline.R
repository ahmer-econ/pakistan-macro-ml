# ============================================================
# Project: Machine Learning Meets Econometrics
# Script 03: OLS Baseline Model
# Author: Ahmer | GitHub: ahmer-econ | August 2026
# ============================================================

setwd("D:/Documents/APPLIED ECONOMETRICS WORK/ML Model")

library(tidyverse)

# ============================================================
# SECTION 1: LOAD DATA
# ============================================================

train <- read.csv("data_clean/train.csv", stringsAsFactors = FALSE)
test  <- read.csv("data_clean/test.csv",  stringsAsFactors = FALSE)
train$date <- as.Date(train$date)
test$date  <- as.Date(test$date)

# Define predictor names
predictors <- c("cpi_lag1","cpi_lag2","cpi_lag3",
                "m2_lag1","m2_lag2",
                "pkr_lag1","pkr_lag2",
                "rate_lag1","rate_lag2",
                "oil_lag1","oil_lag2",
                "fao_lag1","fao_lag2")

# ============================================================
# SECTION 2: FIT OLS
# ============================================================

formula_ols <- as.formula(paste("cpi ~", paste(predictors, collapse = " + ")))
ols_model   <- lm(formula_ols, data = train)
summary(ols_model)
# ============================================================
# SECTION 3: OOS PREDICTION & METRICS
# ============================================================

# Predict on test set
test$ols_pred <- predict(ols_model, newdata = test)

# Compute metrics
ols_rmse <- sqrt(mean((test$cpi - test$ols_pred)^2))
ols_mae  <- mean(abs(test$cpi - test$ols_pred))

cat("OLS OOS RMSE:", round(ols_rmse, 4), "\n")
cat("OLS OOS MAE: ", round(ols_mae,  4), "\n")

# Naive benchmark (last known value = cpi_lag1)
naive_rmse <- sqrt(mean((test$cpi - test$cpi_lag1)^2))
naive_mae  <- mean(abs(test$cpi - test$cpi_lag1))

cat("Naive RMSE:", round(naive_rmse, 4), "\n")
cat("Naive MAE: ", round(naive_mae,  4), "\n")

# ============================================================
# SECTION 4: PLOT OLS PREDICTIONS vs ACTUALS
# ============================================================

ggplot() +
  geom_line(data = train, aes(x = date, y = cpi), color = "grey60", linewidth = 0.6) +
  geom_line(data = test, aes(x = date, y = cpi),      color = "#1565C0", linewidth = 1, linetype = "solid") +
  geom_line(data = test, aes(x = date, y = ols_pred), color = "#D32F2F", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dotted", color = "black") +
  labs(title = "OLS Baseline — Actual vs Predicted CPI (2023 Holdout)",
       subtitle = "Blue = Actual | Red dashed = OLS Predicted | Grey = Training period",
       x = NULL, y = "CPI YoY %") +
  theme_minimal(base_size = 11)

ggsave("outputs/plots/04_ols_predictions.png", width = 10, height = 5, dpi = 150)
cat("Script 03 complete.\n")