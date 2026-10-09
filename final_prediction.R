# ---- SECTION 01: SETUP ----
library(caret)
library(class)
library(rpart)

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

# ---- SECTION 02: TRAIN THE IDENTIFIED MODELS ----
set.seed(123)
tree_full  <- rpart(health_risk ~ ., data = train_data, method = "class",
                    control = rpart.control(cp = 0.001, minsplit = 20, maxdepth = 6))
best_cp    <- tree_full$cptable[which.min(tree_full$cptable[, "xerror"]), "CP"]
tree_model <- prune(tree_full, cp = best_cp)
log_model  <- glm(I(health_risk == "High_Risk") ~ . - chronic_condition,
                  data = train_data, family = binomial)
multi_model <- lm(blood_pressure ~ age + bmi + stress_level + heart_rate + glucose_level +
                    cholesterol + sleep_hours + smoking_status, data = train_data)
best_k <- 15

# ---- SECTION 03: NEW PATIENTS ----
new_patients <- data.frame(
  age = c(28, 52, 68, 45), gender = c("Female", "Male", "Male", "Female"),
  bmi = c(22.4, 31.8, 29.5, 25.0), blood_pressure = c(112, 138, 150, 120),
  heart_rate = c(68, 82, 78, 72), glucose_level = c(86, 118, 160, 92),
  cholesterol = c(170, 235, 250, 190), oxygen_saturation = c(98.5, 96.5, 95.5, 97.8),
  waist_circumference_cm = c(74, 108, 104, 85),
  smoking_status = c("Never", "Current", "Former", "Never"),
  alcohol_consumption = c("None", "Regular", "Occasional", "Occasional"),
  physical_activity = c("High", "Low", "Low", "Moderate"),
  sleep_hours = c(7.5, 5.5, 6.0, 7.0), family_history = c("No", "Yes", "Yes", "No"),
  medication_count = c(0, 2, 5, 0), hospital_visits = c(0, 3, 6, 1),
  stress_level = c(3, 8, 6, 4), exercise_frequency = c(5, 1, 1, 3),
  diet_quality = c("Good", "Poor", "Average", "Average"),
  chronic_condition = c("None", "Hypertension", "Diabetes", "None"),
  healthcare_expense = c(300, 2100, 5200, 600), stringsAsFactors = FALSE)
for (col in names(new_patients)) {
  if (is.factor(data[[col]])) new_patients[[col]] <- factor(new_patients[[col]], levels = levels(data[[col]]))
}
new_patients[, c("age", "gender", "bmi", "blood_pressure", "glucose_level", "smoking_status")]

# ---- SECTION 04: PREDICT THE OUTPUT ----
tree_class <- predict(tree_model, new_patients, type = "class")
log_prob   <- predict(log_model, new_patients, type = "response")
log_class  <- ifelse(log_prob >= 0.5, "High_Risk", "Low_Risk")

features <- setdiff(names(data), "health_risk")
x_all    <- model.matrix(~ . - 1, data = rbind(data[features], new_patients))
x_train  <- x_all[train_index, ]
x_new    <- x_all[-seq_len(nrow(data)), ]
m <- colMeans(x_train); s <- apply(x_train, 2, sd)
knn_class <- knn(scale(x_train, m, s), scale(x_new, m, s),
                 cl = train_data$health_risk, k = best_k)

bp_pred <- predict(multi_model, new_patients)
predictions <- data.frame(Patient = paste0("P", 1:4), Age = new_patients$age,
  KNN = knn_class, Tree = tree_class, Logistic = log_class,
  Risk_Prob = round(log_prob, 3), Pred_BP = round(bp_pred, 1))
predictions
