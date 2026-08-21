# Machine Learning Meets Econometrics — LASSO, Ridge & Elastic Net for Pakistani Inflation Forecasting

Applies penalised regression methods to the problem of variable selection for Pakistani CPI inflation forecasting. A 13-variable candidate predictor matrix of lagged macroeconomic variables is constructed and evaluated against an OLS baseline and naïve benchmark over a 12-month out-of-sample holdout (January–December 2023).

## Key Results

| Model | OOS RMSE | OOS MAE | Predictors Used |
|-------|----------|---------|-----------------|
| Naïve (lag 1) | 27.30 | 13.42 | 1 |
| OLS | 24.88 | 10.81 | 13 |
| Ridge | 28.59 | 14.20 | 13 (shrunk) |
| LASSO | 26.40 | 10.41 | 5 |
| Elastic Net | 26.40 | 10.41 | 5 |

## LASSO Variable Selection

LASSO selects 5 of 13 candidate predictors, zeroing out 8 variables entirely:

**Retained:** CPI lag 1, M2 growth lags 1–2, PKR/USD lag 1, Brent crude lag 2

**Eliminated:** CPI lags 2–3, PKR lag 2, policy rate lags 1–2, oil lag 1, FAO food price lags 1–2

The convergence of LASSO and Elastic Net on an identical variable set provides robustness evidence that these five predictors represent the genuinely informative subset of the candidate matrix.

## Data

| Variable | Source | Frequency |
|----------|--------|-----------|
| CPI Inflation YoY % | IMF IFS | Monthly |
| M2 Broad Money Growth | IMF MFS | Monthly |
| PKR/USD Exchange Rate | State Bank of Pakistan | Monthly |
| Policy Rate | State Bank of Pakistan | Event-based → monthly |
| Brent Crude Oil (USD/bbl) | IMF via FRED | Monthly |
| FAO Food Price Index | FAO | Monthly |

Sample: January 2012 – December 2023 (144 observations)
Training: January 2012 – December 2022 (132 obs)
Holdout: January 2023 – December 2023 (12 obs)

## Scripts

| Script | Purpose |
|--------|---------|
| 01_data_prep.R | Load raw data, clean all six series, construct lag matrix, export train/test CSVs |
| 02_eda.R | CPI time series plot, predictor panel, correlation heatmap |
| 03_ols_baseline.R | OLS with all 13 predictors, OOS RMSE/MAE, residual plot |
| 04_ridge.R | Ridge regression, CV lambda selection, coefficient path, OOS evaluation |
| 05_lasso_elasticnet.R | LASSO and Elastic Net, variable selection, OOS evaluation |
| 06_comparison.R | RMSE bar chart, all-models comparison plot, coefficient table |
| 07_diagnostics.R | Residual diagnostics, CV RMSE, monthly error analysis |

## Software

R 4.6 — packages: tidyverse, glmnet, readxl, lubridate, ggplot2, gridExtra, reshape2

## Author

Ahmer | GitHub: ahmer-econ | August 2026
