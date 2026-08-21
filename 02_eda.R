install.packages("reshape2")
library(reshape2)
# ============================================================
# SECTION 2: TIME SERIES PLOTS OF RAW VARIABLES
# ============================================================

# Create outputs/plots folder if needed
dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)

# Plot 1: CPI Inflation over time
p1 <- ggplot(df, aes(x = date, y = cpi)) +
  geom_line(color = "#2196F3", linewidth = 0.8) +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dashed", color = "red") +
  labs(title = "CPI Inflation YoY % — Pakistan",
       subtitle = "Dashed line = holdout period start",
       x = NULL, y = "YoY %") +
  theme_minimal(base_size = 11)

# Plot 2: All predictors in small multiples
df_long <- df %>%
  select(date, m2_yoy, pkr, policy_rate, brent, food_index) %>%
  pivot_longer(-date, names_to = "variable", values_to = "value") %>%
  mutate(variable = recode(variable,
                           m2_yoy       = "M2 Growth (YoY %)",
                           pkr          = "PKR/USD",
                           policy_rate  = "Policy Rate (%)",
                           brent        = "Brent Crude (USD/bbl)",
                           food_index   = "FAO Food Price Index"
  ))

p2 <- ggplot(df_long, aes(x = date, y = value)) +
  geom_line(color = "#2196F3", linewidth = 0.6) +
  geom_vline(xintercept = as.Date("2023-01-01"), linetype = "dashed", color = "red") +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  labs(title = "Candidate Predictors — Pakistan Macro Variables",
       subtitle = "Dashed line = holdout period start",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 10)

# Plot 3: Correlation heatmap of predictor matrix (training set only)
library(reshape2)
cor_vars <- train %>%
  select(cpi, cpi_lag1, cpi_lag2, cpi_lag3,
         m2_lag1, m2_lag2, pkr_lag1, pkr_lag2,
         rate_lag1, rate_lag2, oil_lag1, oil_lag2,
         fao_lag1, fao_lag2)

cor_matrix <- cor(cor_vars, use = "complete.obs")
cor_melt   <- melt(cor_matrix)

p3 <- ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#D32F2F", mid = "white", high = "#1565C0",
                       midpoint = 0, limits = c(-1, 1), name = "r") +
  geom_text(aes(label = round(value, 2)), size = 2.5) +
  labs(title = "Correlation Matrix — Predictor Variables (Training Set)",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Save all plots
ggsave("outputs/plots/01_cpi_timeseries.png",    p1, width = 10, height = 4, dpi = 150)
ggsave("outputs/plots/02_predictors_panel.png",  p2, width = 12, height = 8, dpi = 150)
ggsave("outputs/plots/03_correlation_heatmap.png", p3, width = 10, height = 8, dpi = 150)

cat("Section 2 complete. Plots saved.\n")