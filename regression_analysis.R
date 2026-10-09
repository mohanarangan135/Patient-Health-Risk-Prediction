# ---- SECTION 01: SETUP ----
library(caret)
library(pROC)
library(ggplot2)

data <- read.csv(file.choose())
for (col in names(data)) {
  if (anyNA(data[[col]])) {
    if (is.numeric(data[[col]])) data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    else data[[col]][is.na(data[[col]])] <- names(which.max(table(data[[col]])))
  }
}
data <- data[!duplicated(data), ]
data$patient_id  <- NULL
data$health_risk <- factor(data$health_risk, levels = c("High_Risk", "Low_Risk"))
char_cols <- sapply(data, is.character)
data[char_cols] <- lapply(data[char_cols], as.factor)
set.seed(123)
train_index <- createDataPartition(data$health_risk, p = 0.8, list = FALSE)
train_data  <- data[train_index, ]
test_data   <- data[-train_index, ]
dim(train_data)
dim(test_data)

# ---- SECTION 02: DATA STRUCTURES ----
ages <- data$age[1:5]                      # vector
ages
class(ages)
bp_matrix <- as.matrix(data[1:3, c("age", "bmi", "blood_pressure")])   # matrix
bp_matrix
patient <- list(age = data$age[1], gender = as.character(data$gender[1]),
                visits = c(2, 0, 1))       # list
str(patient)
levels(data$health_risk)                   # factor
class(data)                                # data frame
dim(data)

# ---- SECTION 03: CONTROL STRUCTURES ----
bmi_value <- data$bmi[1]
if (bmi_value >= 30) {
  print("Obese")
} else if (bmi_value >= 25) {
  print("Overweight")
} else {
  print("Normal or below")
}
for (col in c("age", "bmi", "blood_pressure", "glucose_level")) {
  cat(col, "mean =", round(mean(data[[col]]), 2), "\n")
}
i <- 1; count <- 0
while (count < 3) {
  if (data$health_risk[i] == "High_Risk") count <- count + 1
  i <- i + 1
}
cat("First 3 High_Risk patients found within", i - 1, "records\n")

# ---- SECTION 04: FUNCTIONS ----
bmi_category <- function(bmi) {
  ifelse(bmi < 18.5, "Underweight",
  ifelse(bmi < 25, "Normal",
  ifelse(bmi < 30, "Overweight", "Obese")))
}
bmi_group <- bmi_category(data$bmi)
table(bmi_group)
rmse <- function(actual, predicted) sqrt(mean((actual - predicted)^2))
num_cols <- sapply(data, is.numeric)
round(sapply(data[num_cols][1:5], mean), 2)

# ---- SECTION 05: LISTS AND DATA FRAMES ----
data_info <- list(rows = nrow(data), columns = ncol(data),
                  target_levels = levels(data$health_risk),
                  numeric_columns = sum(num_cols))
str(data_info)
high_risk <- subset(data, health_risk == "High_Risk")
nrow(high_risk)
aggregate(cbind(age, bmi, glucose_level) ~ health_risk, data = data, FUN = mean)
head(data[order(-data$healthcare_expense), c("age", "chronic_condition", "healthcare_expense")], 3)

# ---- SECTION 06: OUTLIERS AND CORRELATION ----
count_outliers <- function(x) {
  q <- quantile(x, c(0.25, 0.75)); iqr <- q[2] - q[1]
  sum(x < q[1] - 1.5 * iqr | x > q[2] + 1.5 * iqr)
}
sapply(data[c("bmi", "blood_pressure", "glucose_level", "cholesterol", "healthcare_expense")], count_outliers)
round(cor(data[c("age", "bmi", "blood_pressure", "glucose_level", "cholesterol", "waist_circumference_cm")]), 2)
ggplot(data, aes(x = healthcare_expense)) +
  geom_histogram(bins = 40, fill = "#2E86C1", colour = "white") +
  labs(title = "Distribution of Healthcare Expense", x = "Healthcare Expense (USD)", y = "Patients") +
  theme_minimal()

# ---- SECTION 07: HYPOTHESIS TESTS ----
t.test(glucose_level ~ health_risk, data = data)
chisq.test(table(data$smoking_status, data$health_risk))

# ---- SECTION 08: SIMPLE LINEAR REGRESSION ----
simple_model <- lm(blood_pressure ~ age, data = train_data)
summary(simple_model)
simple_pred <- predict(simple_model, test_data)
cat("Test R2 =", round(cor(test_data$blood_pressure, simple_pred)^2, 4),
    " RMSE =", round(rmse(test_data$blood_pressure, simple_pred), 2), "\n")
ggplot(train_data, aes(x = age, y = blood_pressure)) +
  geom_point(alpha = 0.35, colour = "#2E86C1") +
  geom_smooth(method = "lm", colour = "#C0392B", se = FALSE) +
  labs(title = "Simple Linear Regression: Blood Pressure vs Age", x = "Age", y = "Blood Pressure") +
  theme_minimal()

# ---- SECTION 09: MULTIPLE LINEAR REGRESSION ----
multi_model <- lm(blood_pressure ~ age + bmi + stress_level + heart_rate + glucose_level +
                    cholesterol + sleep_hours + smoking_status, data = train_data)
summary(multi_model)
multi_pred <- predict(multi_model, test_data)
cat("Test R2 =", round(cor(test_data$blood_pressure, multi_pred)^2, 4),
    " RMSE =", round(rmse(test_data$blood_pressure, multi_pred), 2), "\n")

# ---- SECTION 10: STEPWISE REGRESSION ----
full_model <- lm(blood_pressure ~ . - health_risk - chronic_condition, data = train_data)
step_model <- step(full_model, direction = "both", trace = 0)
formula(step_model)
step_pred <- predict(step_model, test_data)
cat("Test R2 =", round(cor(test_data$blood_pressure, step_pred)^2, 4),
    " RMSE =", round(rmse(test_data$blood_pressure, step_pred), 2), "\n")
ggplot(data.frame(Actual = test_data$blood_pressure, Predicted = step_pred),
       aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.4, colour = "#2E86C1") +
  geom_abline(slope = 1, intercept = 0, colour = "#C0392B") +
  labs(title = "Stepwise Model: Actual vs Predicted Blood Pressure") +
  theme_minimal()

# ---- SECTION 11: LOGISTIC REGRESSION ----
log_model <- glm(I(health_risk == "High_Risk") ~ . - chronic_condition, data = train_data, family = binomial)
log_prob  <- predict(log_model, test_data, type = "response")
log_pred  <- factor(ifelse(log_prob >= 0.5, "High_Risk", "Low_Risk"), levels = c("High_Risk", "Low_Risk"))
log_cm <- confusionMatrix(log_pred, test_data$health_risk, positive = "High_Risk")
log_cm
log_roc <- roc(test_data$health_risk, log_prob, levels = c("Low_Risk", "High_Risk"), direction = "<")
auc(log_roc)

# ---- SECTION 12: REGRESSION MODEL COMPARISON ----
get_r2 <- function(p) cor(test_data$blood_pressure, p)^2
get_mae <- function(p) mean(abs(test_data$blood_pressure - p))
reg_compare <- data.frame(
  Model = c("Simple Linear", "Multiple Linear", "Stepwise"),
  R2   = c(get_r2(simple_pred), get_r2(multi_pred), get_r2(step_pred)),
  RMSE = c(rmse(test_data$blood_pressure, simple_pred), rmse(test_data$blood_pressure, multi_pred), rmse(test_data$blood_pressure, step_pred)),
  MAE  = c(get_mae(simple_pred), get_mae(multi_pred), get_mae(step_pred)))
reg_compare[, -1] <- round(reg_compare[, -1], 4)
reg_compare
