dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)
# Project: Machine Learning Meets Econometrics
# Script 06: Model Comparison and Final Results
# Author: Ahmer | GitHub: ahmer-econ | August 2026

setwd("D:/Documents/APPLIED ECONOMETRICS WORK/ML Model")

library(tidyverse)
library(glmnet)

# SECTION 1: LOAD DATA AND REFIT ALL MODELS

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

# OLS
formula_ols <- as.formula(paste("cpi ~", paste(predictors, collapse = " + ")))
ols_model   <- lm(formula_ols, data = train)
test$ols_pred <- predict(ols_model, newdata = test)

# Ridge
set.seed(42)
ridge_cv    <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10, standardize = TRUE)
ridge_model <- glmnet(X_train, y_train, alpha = 0, lambda = ridge_cv$lambda.min, standardize = TRUE)
test$ridge_pred <- as.numeric(predict(ridge_model, newx = X_test))

# LASSO
set.seed(42)
lasso_cv    <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10, standardize = TRUE)
lasso_model <- glmnet(X_train, y_train, alpha = 1, lambda = lasso_cv$lambda.min, standardize = TRUE)
test$lasso_pred <- as.numeric(predict(lasso_model, newx = X_test))

# Elastic Net
set.seed(42)
en_cv    <- cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = 10, standardize = TRUE)
en_model <- glmnet(X_train, y_train, alpha = 0.5, lambda = en_cv$lambda.min, standardize = TRUE)
test$en_pred <- as.numeric(predict(en_model, newx = X_test))

# Naive
test$naive_pred <- test$cpi_lag1

# SECTION 2: RESULTS TABLE

results <- data.frame(
  Model = c("Naive","OLS","Ridge","LASSO","Elastic Net"),
  RMSE  = c(
    sqrt(mean((y_test - test$naive_pred)^2)),
    sqrt(mean((y_test - test$ols_pred)^2)),
    sqrt(mean((y_test - test$ridge_pred)^2)),
    sqrt(mean((y_test - test$lasso_pred)^2)),
    sqrt(mean((y_test - test$en_pred)^2))
  ),
  MAE   = c(
    mean(abs(y_test - test$naive_pred)),
    mean(abs(y_test - test$ols_pred)),
    mean(abs(y_test - test$ridge_pred)),
    mean(abs(y_test - test$lasso_pred)),
    mean(abs(y_test - test$en_pred))
  )
)

results$RMSE <- round(results$RMSE, 4)
results$MAE  <- round(results$MAE,  4)
print(results)

write.csv(results, "outputs/tables/model_comparison.csv", row.names = FALSE)

# SECTION 3: RMSE BAR CHART

dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)

p_rmse <- ggplot(results, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = round(RMSE, 2)), hjust = -0.2, size = 3.5) +
  scale_fill_manual(values = c(
    "Naive"       = "#B0BEC5",
    "OLS"         = "#1565C0",
    "Ridge"       = "#E65100",
    "LASSO"       = "#2E7D32",
    "Elastic Net" = "#6A1B9A"
  )) +
  coord_flip() +
  ylim(0, 35) +
  labs(title = "OOS RMSE Comparison — All Models",
       subtitle = "Lower is better | 2023 holdout (12 months)",
       x = NULL, y = "RMSE") +
  theme_minimal(base_size = 11)

ggsave("outputs/plots/12_rmse_comparison.png", p_rmse, width = 8, height = 5, dpi = 150)

# SECTION 4: ALL PREDICTIONS ON ONE PLOT

p_all <- ggplot() +
  geom_line(data = train, aes(x = date, y = cpi), color = "grey70", linewidth = 0.5) +
  geom_line(data = test, aes(x = date, y = cpi),        color = "black",   linewidth = 1.2) +
  geom_line(data = test, aes(x = date, y = ols_pred),   color = "#1565C0", linewidth = 0.8, linetype = "dashed") +
  geom_line(data = test, aes(x = date, y = ridge_pred), color = "#E65100", linewidth = 0.8, linetype = "dashed") +
  geom_line(data = test, aes(x = date, y = lasso_pred), color = "#2E7D32", linewidth = 0.8, linetype = "dashed") +
  geom_line(data = test, aes(x = date, y = en_pred),    color = "#6A1B9A", linewidth = 0.8, linetype = "dashed") +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dotted") +
  annotate("text", x = as.Date("2023-02-01"), y = 20,
           label = "OLS=blue | Ridge=orange\nLASSO=green | EN=purple",
           hjust = 0, size = 3, color = "grey30") +
  labs(title = "All Models vs Actual CPI — 2023 Holdout",
       subtitle = "Black = Actual | Dashed = Model predictions | Grey = Training period",
       x = NULL, y = "CPI YoY %") +
  theme_minimal(base_size = 11)

ggsave("outputs/plots/13_all_models_comparison.png", p_all, width = 11, height = 5, dpi = 150)

# SECTION 5: LASSO COEFFICIENT PLOT

lasso_coef_df <- as.data.frame(as.matrix(coef(lasso_model)))
lasso_coef_df$variable <- rownames(lasso_coef_df)
names(lasso_coef_df)[1] <- "coefficient"
lasso_coef_df <- lasso_coef_df %>%
  filter(variable != "(Intercept)", coefficient != 0)

p_coef <- ggplot(lasso_coef_df, aes(x = reorder(variable, coefficient), y = coefficient, fill = coefficient > 0)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  scale_fill_manual(values = c("TRUE" = "#1565C0", "FALSE" = "#D32F2F")) +
  coord_flip() +
  labs(title = "LASSO Selected Variables — Coefficients",
       subtitle = "Only non-zero coefficients shown | Blue = positive | Red = negative",
       x = NULL, y = "Coefficient") +
  theme_minimal(base_size = 11)

ggsave("outputs/plots/14_lasso_coefficients.png", p_coef, width = 8, height = 5, dpi = 150)

cat("Script 06 complete. All outputs saved.\n")