# ============================================================
# Project: Machine Learning Meets Econometrics
# Script 01: Data Preparation & Predictor Matrix Construction
# Author: Ahmer | GitHub: ahmer-econ | August 2026
# ============================================================

setwd("D:/Documents/APPLIED ECONOMETRICS WORK/ML Model")

library(tidyverse)
library(readxl)
library(lubridate)

# ============================================================
# SECTION 1: LOAD RAW DATA
# ============================================================

cpi_raw  <- read.csv("data_raw/CPI Inflation.csv.csv",        stringsAsFactors = FALSE)
m2_raw   <- read.csv("data_raw/Monetory Aggregates csv.csv",  stringsAsFactors = FALSE)
pkr_raw2 <- read_excel("data_raw/Bank Floating Average Exchange Rates.xls.xls", col_names = FALSE)
rate_raw <- read.csv("data_raw/Policy Rate.csv.csv",          stringsAsFactors = FALSE)
oil_raw  <- read.csv("data_raw/Brent_Crude_FRED.csv.csv",     stringsAsFactors = FALSE)
fao_raw  <- read.csv("data_raw/food_price.csv.csv",           stringsAsFactors = FALSE)

# ============================================================
# SECTION 2: CLEAN EACH SERIES
# ============================================================

# --- 2a. CPI (row 26, IMF wide format) ---
cpi_row  <- cpi_raw[26, ]
cpi_cols <- names(cpi_raw)[grepl("^X\\d{4}\\.M", names(cpi_raw))]
cpi_vals <- as.numeric(cpi_row[, cpi_cols])
cpi_dates <- as.Date(paste0(sub("X(\\d{4})\\.M(\\d{2})", "\\1-\\2-01", cpi_cols)), format = "%Y-%m-%d")
cpi_clean <- data.frame(date = cpi_dates, cpi = cpi_vals) %>%
  filter(!is.na(cpi)) %>% arrange(date)

# --- 2b. M2 Broad Money ---
m2_row_num <- which(m2_raw$SERIES_CODE == "PAK.DCORP_L_BM.XDC.M")
m2_row     <- m2_raw[m2_row_num, ]
m2_cols    <- names(m2_raw)[grepl("^X\\d{4}\\.M", names(m2_raw))]
m2_vals    <- as.numeric(m2_row[, m2_cols])
m2_dates   <- as.Date(paste0(sub("X(\\d{4})\\.M(\\d{2})", "\\1-\\2-01", m2_cols)), format = "%Y-%m-%d")
m2_clean   <- data.frame(date = m2_dates, m2_level = m2_vals) %>%
  filter(!is.na(m2_level)) %>% arrange(date) %>%
  mutate(m2_yoy = (m2_level / lag(m2_level, 12) - 1) * 100)

# --- 2c. PKR/USD Exchange Rate ---
pkr_data3 <- pkr_raw2[14:nrow(pkr_raw2), c(1, 2, 24)]
names(pkr_data3) <- c("month", "year", "pkr")
pkr_clean <- pkr_data3 %>%
  mutate(year = as.character(year)) %>%
  tidyr::fill(year, .direction = "down") %>%
  mutate(month = as.character(month), pkr = as.numeric(pkr)) %>%
  filter(!is.na(pkr), grepl("^\\d{4}$", year),
         month %in% c("Jan","Feb","Mar","Apr","May","Jun",
                      "Jul","Aug","Sep","Oct","Nov","Dec",
                      "January","February","March","April","May","June",
                      "July","August","September","October","November","December")) %>%
  mutate(date = as.Date(paste(year, month, "01"), format = "%Y %B %d"),
         date = if_else(is.na(date),
                        as.Date(paste(year, month, "01"), format = "%Y %b %d"), date)) %>%
  select(date, pkr) %>% filter(!is.na(date)) %>% arrange(date)

# --- 2d. Policy Rate (Target Rate: SBPOL0030) ---
rate_clean <- rate_raw %>%
  filter(grepl("SBPOL0030", Series.Key)) %>%
  mutate(date = as.Date(Observation.Date, format = "%d-%b-%Y"),
         policy_rate = as.numeric(Observation.Value)) %>%
  select(date, policy_rate) %>%
  filter(!is.na(date), !is.na(policy_rate)) %>% arrange(date)

# --- 2e. Brent Crude Oil ---
oil_clean <- oil_raw %>%
  mutate(date = as.Date(observation_date, format = "%m/%d/%Y"),
         brent = POILBREUSDM_20260713) %>%
  select(date, brent) %>% filter(!is.na(brent)) %>% arrange(date)

# --- 2f. FAO Food Price Index ---
fao_clean <- fao_raw %>%
  rename(date_str = 1, food_index = 2) %>%
  mutate(date = as.Date(paste0(date_str, "-01"), format = "%Y-%m-%d")) %>%
  select(date, food_index) %>%
  filter(!is.na(date), !is.na(food_index)) %>% arrange(date)

# ============================================================
# SECTION 3: MERGE
# ============================================================

date_spine <- data.frame(
  date = seq(as.Date("2010-01-01"), as.Date("2024-12-01"), by = "month")
)

rate_monthly <- date_spine %>%
  left_join(rate_clean, by = "date") %>%
  tidyr::fill(policy_rate, .direction = "down")

df <- date_spine %>%
  left_join(cpi_clean, by = "date") %>%
  left_join(m2_clean %>% select(date, m2_yoy), by = "date") %>%
  left_join(pkr_clean, by = "date") %>%
  left_join(rate_monthly, by = "date") %>%
  left_join(oil_clean, by = "date") %>%
  left_join(fao_clean, by = "date") %>%
  tidyr::fill(policy_rate, .direction = "updown")

# ============================================================
# SECTION 4: LAG MATRIX
# ============================================================

df_sample <- df %>%
  filter(date >= as.Date("2011-01-01") & date <= as.Date("2023-12-01"))

df_lags <- df_sample %>%
  mutate(
    cpi_lag1  = lag(cpi, 1), cpi_lag2  = lag(cpi, 2), cpi_lag3  = lag(cpi, 3),
    m2_lag1   = lag(m2_yoy, 1), m2_lag2   = lag(m2_yoy, 2),
    pkr_lag1  = lag(pkr, 1), pkr_lag2  = lag(pkr, 2),
    rate_lag1 = lag(policy_rate, 1), rate_lag2 = lag(policy_rate, 2),
    oil_lag1  = lag(brent, 1), oil_lag2  = lag(brent, 2),
    fao_lag1  = lag(food_index, 1), fao_lag2  = lag(food_index, 2)
  ) %>%
  filter(date >= as.Date("2012-01-01"))

# ============================================================
# SECTION 5: EXPORT
# ============================================================

train <- df_lags %>% filter(date <= as.Date("2022-12-01"))
test  <- df_lags %>% filter(date >= as.Date("2023-01-01"))

write.csv(df_lags, "data_clean/predictor_matrix.csv", row.names = FALSE)
write.csv(train,   "data_clean/train.csv",             row.names = FALSE)
write.csv(test,    "data_clean/test.csv",              row.names = FALSE)

cat("Script 01 complete.\n")
cat("Predictor matrix:", nrow(df_lags), "rows,", ncol(df_lags), "cols\n")
cat("Train:", nrow(train), "| Test:", nrow(test), "\n")
cat("NAs:", sum(colSums(is.na(df_lags))), "\n")