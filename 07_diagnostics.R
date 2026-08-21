# Project: Machine Learning Meets Econometrics
# Script 07: Diagnostics and Extended Analysis
# Author: Ahmer | GitHub: ahmer-econ | August 2026

setwd("D:/Documents/APPLIED ECONOMETRICS WORK/ML Model")

library(tidyverse)
library(glmnet)

# SECTION 1: LOAD AND REFIT ALL MODELS

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

formula_ols <- as.formula(paste("cpi ~", paste(predictors, collapse = " + ")))
ols_model   <- lm(formula_ols, data = train)
train$ols_fitted   <- fitted(ols_model)
train$ols_resid    <- residuals(ols_model)
test$ols_pred      <- predict(ols_model, newdata = test)

set.seed(42)
ridge_cv    <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10, standardize = TRUE)
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = ridge_cv$lambda.min, standardize = TRUE)
test$ridge_pred <- as.numeric(predict(ridge_model, newx = X_test))

set.seed(42)
lasso_cv    <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10, standardize = TRUE)
lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lasso_cv$lambda.min, standardize = TRUE)
test$lasso_pred <- as.numeric(predict(lasso_model, newx = X_test))

set.seed(42)
en_cv    <- cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = 10, standardize = TRUE)
en_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = en_cv$lambda.min, standardize = TRUE)
test$en_pred <- as.numeric(predict(en_model, newx = X_test))

test$naive_pred <- test$cpi_lag1

# SECTION 2: OLS RESIDUAL DIAGNOSTICS

# Plot 1: Residuals vs Fitted
p_resfit <- ggplot(train, aes(x = ols_fitted, y = ols_resid)) +
  geom_point(color = "#1565C0", alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = FALSE, color = "orange", linewidth = 0.8) +
  labs(title = "OLS Residuals vs Fitted Values",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal(base_size = 11)

# Plot 2: Residuals over time
p_restime <- ggplot(train, aes(x = date, y = ols_resid)) +
  geom_line(color = "#1565C0", linewidth = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_hline(yintercept = c(-2*sd(train$ols_resid), 2*sd(train$ols_resid)),
             linetype = "dotted", color = "grey50") +
  labs(title = "OLS Residuals Over Time",
       subtitle = "Dotted lines = +/- 2 standard deviations",
       x = NULL, y = "Residuals") +
  theme_minimal(base_size = 11)

# Plot 3: Histogram of residuals
p_reshist <- ggplot(train, aes(x = ols_resid)) +
  geom_histogram(aes(y = after_stat(density)), bins = 25,
                 fill = "#1565C0", alpha = 0.7, color = "white") +
  geom_density(color = "red", linewidth = 0.8) +
  labs(title = "Distribution of OLS Residuals",
       x = "Residuals", y = "Density") +
  theme_minimal(base_size = 11)

ggsave("outputs/plots/15_resid_vs_fitted.png",  p_resfit,  width = 8, height = 5, dpi = 150)
ggsave("outputs/plots/16_resid_over_time.png",  p_restime, width = 10, height = 4, dpi = 150)
ggsave("outputs/plots/17_resid_histogram.png",  p_reshist, width = 7, height = 5, dpi = 150)
cat("Residual plots saved.\n")

# SECTION 3: LAMBDA SENSITIVITY — RIDGE

cat("\nRidge lambda.min:", round(ridge_cv$lambda.min, 4))
cat("\nRidge lambda.1se:", round(ridge_cv$lambda.1se, 4))

ridge_model_1se <- glmnet(X_train, y_train, alpha = 0,
                          lambda = ridge_cv$lambda.1se, standardize = TRUE)
pred_1se <- as.numeric(predict(ridge_model_1se, newx = X_test))
rmse_1se <- sqrt(mean((y_test - pred_1se)^2))
rmse_min <- sqrt(mean((y_test - test$ridge_pred)^2))

cat("\nRidge RMSE (lambda.min):", round(rmse_min, 4))
cat("\nRidge RMSE (lambda.1se):", round(rmse_1se, 4), "\n")

# SECTION 4: CV RMSE ON TRAINING SET

cat("\nCV RMSE on Training Set (10-fold):\n")
cat("Ridge  CV RMSE:", round(sqrt(min(ridge_cv$cvm)), 4), "\n")
cat("LASSO  CV RMSE:", round(sqrt(min(lasso_cv$cvm)), 4), "\n")
cat("EN     CV RMSE:", round(sqrt(min(en_cv$cvm)),    4), "\n")

# SECTION 5: COEFFICIENT COMPARISON TABLE

ols_coefs   <- coef(ols_model)[predictors]
ridge_coefs <- as.numeric(coef(ridge_model))[2:14]
lasso_coefs <- as.numeric(coef(lasso_model))[2:14]
en_coefs    <- as.numeric(coef(en_model))[2:14]

coef_table <- data.frame(
  Variable    = predictors,
  OLS         = round(ols_coefs,   4),
  Ridge       = round(ridge_coefs, 4),
  LASSO       = round(lasso_coefs, 4),
  ElasticNet  = round(en_coefs,    4)
)
print(coef_table)
write.csv(coef_table, "outputs/tables/coefficient_comparison.csv", row.names = FALSE)
cat("Coefficient table saved.\n")

# SECTION 6: PREDICTION ERROR BY MONTH

error_table <- test %>%
  select(date, cpi, ols_pred, ridge_pred, lasso_pred, en_pred, naive_pred) %>%
  mutate(
    month      = format(date, "%b %Y"),
    err_ols    = round(cpi - ols_pred,   2),
    err_ridge  = round(cpi - ridge_pred, 2),
    err_lasso  = round(cpi - lasso_pred, 2),
    err_en     = round(cpi - en_pred,    2),
    err_naive  = round(cpi - naive_pred, 2)
  ) %>%
  select(month, cpi, err_naive, err_ols, err_ridge, err_lasso, err_en)

print(error_table)
write.csv(error_table, "outputs/tables/monthly_errors.csv", row.names = FALSE)

# Plot: Monthly prediction errors
error_long <- error_table %>%
  pivot_longer(cols = starts_with("err_"),
               names_to = "model", values_to = "error") %>%
  mutate(model = recode(model,
                        err_naive = "Naive", err_ols = "OLS",
                        err_ridge = "Ridge", err_lasso = "LASSO", err_en = "Elastic Net"
  ))

p_errors <- ggplot(error_long, aes(x = month, y = error, color = model, group = model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  scale_color_manual(values = c(
    "Naive" = "#B0BEC5", "OLS" = "#1565C0", "Ridge" = "#E65100",
    "LASSO" = "#2E7D32", "Elastic Net" = "#6A1B9A"
  )) +
  labs(title = "Monthly Prediction Errors by Model — 2023 Holdout",
       subtitle = "Error = Actual minus Predicted | Positive = underprediction",
       x = NULL, y = "Error (pp)", color = NULL) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

ggsave("outputs/plots/18_monthly_errors.png", p_errors, width = 11, height = 5, dpi = 150)
cat("Script 07 complete.\n")