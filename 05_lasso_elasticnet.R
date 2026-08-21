# Project: Machine Learning Meets Econometrics
# Script 05: LASSO and Elastic Net Regression
# Author: Ahmer | GitHub: ahmer-econ | August 2026

setwd("D:/Documents/APPLIED ECONOMETRICS WORK/ML Model")

library(tidyverse)
library(glmnet)

# SECTION 1: LOAD DATA

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

# SECTION 2: LASSO

set.seed(42)
lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10, standardize = TRUE)
best_lambda_lasso <- lasso_cv$lambda.min
cat("Best lambda LASSO:", round(best_lambda_lasso, 4), "\n")

lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda_lasso, standardize = TRUE)
print(coef(lasso_model))

# SECTION 3: ELASTIC NET

set.seed(42)
en_cv <- cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = 10, standardize = TRUE)
best_lambda_en <- en_cv$lambda.min
cat("Best lambda Elastic Net:", round(best_lambda_en, 4), "\n")

en_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda_en, standardize = TRUE)
print(coef(en_model))

# SECTION 4: OOS METRICS

test$lasso_pred <- as.numeric(predict(lasso_model, newx = X_test))
test$en_pred    <- as.numeric(predict(en_model,    newx = X_test))

lasso_rmse <- sqrt(mean((y_test - test$lasso_pred)^2))
lasso_mae  <- mean(abs(y_test - test$lasso_pred))
en_rmse    <- sqrt(mean((y_test - test$en_pred)^2))
en_mae     <- mean(abs(y_test - test$en_pred))

cat("LASSO RMSE:", round(lasso_rmse, 4), "\n")
cat("LASSO MAE: ", round(lasso_mae,  4), "\n")
cat("EN RMSE:",    round(en_rmse,    4), "\n")
cat("EN MAE:  ",   round(en_mae,     4), "\n")

# SECTION 5: PLOTS

png("outputs/plots/08_lasso_cv_lambda.png", width = 800, height = 500, res = 120)
plot(lasso_cv)
title("LASSO 10-Fold CV: MSE vs Log Lambda", line = 2.5)
dev.off()

lasso_path <- glmnet(X_train, y_train, alpha = 1, standardize = TRUE)
png("outputs/plots/09_lasso_coef_path.png", width = 800, height = 500, res = 120)
plot(lasso_path, xvar = "lambda", label = TRUE)
title("LASSO Coefficient Paths", line = 2.5)
dev.off()

p_lasso <- ggplot() +
  geom_line(data = train, aes(x = date, y = cpi), color = "grey60", linewidth = 0.6) +
  geom_line(data = test, aes(x = date, y = cpi), color = "#1565C0", linewidth = 1) +
  geom_line(data = test, aes(x = date, y = lasso_pred), color = "#2E7D32", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dotted") +
  labs(title = "LASSO Actual vs Predicted CPI 2023 Holdout",
       subtitle = "Blue = Actual | Green dashed = LASSO | Grey = Training",
       x = NULL, y = "CPI YoY %") +
  theme_minimal(base_size = 11)

p_en <- ggplot() +
  geom_line(data = train, aes(x = date, y = cpi), color = "grey60", linewidth = 0.6) +
  geom_line(data = test, aes(x = date, y = cpi), color = "#1565C0", linewidth = 1) +
  geom_line(data = test, aes(x = date, y = en_pred), color = "#6A1B9A", linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dotted") +
  labs(title = "Elastic Net Actual vs Predicted CPI 2023 Holdout",
       subtitle = "Blue = Actual | Purple dashed = EN | Grey = Training",
       x = NULL, y = "CPI YoY %") +
  theme_minimal(base_size = 11)

ggsave("outputs/plots/10_lasso_predictions.png", p_lasso, width = 10, height = 5, dpi = 150)
ggsave("outputs/plots/11_en_predictions.png",    p_en,    width = 10, height = 5, dpi = 150)

cat("Script 05 complete.\n")