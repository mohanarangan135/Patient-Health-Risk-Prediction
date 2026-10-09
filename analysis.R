# ---- SECTION 01: LOAD LIBRARIES AND DATASET ----
library(caret)
library(class)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(pROC)

data <- read.csv(file.choose())
dim(data)

# ---- SECTION 02: STRUCTURE OF DATASET ----
str(data)

# ---- SECTION 03: SUMMARY AND HEAD ----
summary(data)
head(data)

# ---- SECTION 04: MISSING VALUES ----
colSums(is.na(data))

# Handle missing values: median for numeric, mode for categorical
for (col in names(data)) {
  if (anyNA(data[[col]])) {
    if (is.numeric(data[[col]])) {
      data[[col]][is.na(data[[col]])] <- median(data[[col]], na.rm = TRUE)
    } else {
      mode_val <- names(which.max(table(data[[col]])))
      data[[col]][is.na(data[[col]])] <- mode_val
    }
  }
}
sum(is.na(data))

# ---- SECTION 05: DUPLICATES ----
sum(duplicated(data))
data <- data[!duplicated(data), ]
sum(duplicated(data))
dim(data)

# ---- SECTION 06: VISUALIZATION ----
table(data$health_risk)
ggplot(data, aes(x = health_risk, fill = health_risk)) +
  geom_bar(width = 0.55) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("High_Risk" = "#C0392B", "Low_Risk" = "#2E86C1")) +
  labs(title = "Health Risk Distribution", x = "Health Risk", y = "Number of Patients") +
  theme_minimal() + theme(legend.position = "none")

# ---- SECTION 07: DATA PREPARATION ----
data$health_risk <- factor(data$health_risk, levels = c("High_Risk", "Low_Risk"))
data$patient_id  <- NULL
char_cols <- sapply(data, is.character)
data[char_cols] <- lapply(data[char_cols], as.factor)
str(data)

# ---- SECTION 08: DATA SPLITTING ----
set.seed(123)
train_index <- createDataPartition(data$health_risk, p = 0.8, list = FALSE)
train_data  <- data[train_index, ]
test_data   <- data[-train_index, ]
dim(train_data)
dim(test_data)
prop.table(table(train_data$health_risk))
prop.table(table(test_data$health_risk))

# ---- SECTION 09: KNN DATA PREPARATION AND K SELECTION ----
x_all   <- model.matrix(health_risk ~ . - 1, data = data)
x_train <- x_all[train_index, ]
x_test  <- x_all[-train_index, ]
train_mean <- colMeans(x_train)
train_sd   <- apply(x_train, 2, sd)
x_train_s  <- scale(x_train, center = train_mean, scale = train_sd)
x_test_s   <- scale(x_test,  center = train_mean, scale = train_sd)
y_train <- train_data$health_risk
y_test  <- test_data$health_risk

# validation set taken from the training data (80/20) to choose K
set.seed(123)
val_index <- createDataPartition(y_train, p = 0.8, list = FALSE)
k_values  <- seq(3, 21, by = 2)
k_acc <- sapply(k_values, function(k) {
  pred <- knn(x_train_s[val_index, ], x_train_s[-val_index, ],
              cl = y_train[val_index], k = k)
  mean(pred == y_train[-val_index])
})
k_results <- data.frame(K = k_values, Accuracy = k_acc)
print(k_results)
best_k <- k_results$K[which.max(k_results$Accuracy)]
best_k

# ---- SECTION 10: KNN ACCURACY CHART ----
ggplot(k_results, aes(x = K, y = Accuracy)) +
  geom_line(colour = "#2E86C1") + geom_point(size = 3, colour = "#2E86C1") +
  geom_point(data = k_results[k_results$K == best_k, ], colour = "#C0392B", size = 5) +
  scale_x_continuous(breaks = k_values) +
  labs(title = paste("KNN: K vs Validation Accuracy (Best K =", best_k, ")"),
       x = "K value", y = "Validation Accuracy") +
  theme_minimal()

# ---- SECTION 11: DECISION TREE TRAINING ----
set.seed(123)
tree_full <- rpart(health_risk ~ ., data = train_data, method = "class",
                   control = rpart.control(cp = 0.001, minsplit = 20, maxdepth = 6))
printcp(tree_full)
best_cp <- tree_full$cptable[which.min(tree_full$cptable[, "xerror"]), "CP"]
best_cp
tree_model <- prune(tree_full, cp = best_cp)

# ---- SECTION 12: DECISION TREE PLOT ----
rpart.plot(tree_model, type = 2, extra = 104, fallen.leaves = TRUE,
           main = "Decision Tree - Patient Health Risk")
tree_model$variable.importance

# ---- SECTION 13: DECISION TREE EVALUATION ----
tree_pred <- predict(tree_model, test_data, type = "class")
tree_prob <- predict(tree_model, test_data, type = "prob")[, "High_Risk"]
tree_cm <- confusionMatrix(tree_pred, y_test, positive = "High_Risk")
tree_cm
tree_roc <- roc(y_test, tree_prob, levels = c("Low_Risk", "High_Risk"), direction = "<")
auc(tree_roc)

# ---- SECTION 14: KNN EVALUATION ----
set.seed(123)
knn_pred <- knn(x_train_s, x_test_s, cl = y_train, k = best_k, prob = TRUE)
knn_win  <- attr(knn_pred, "prob")
knn_prob <- ifelse(knn_pred == "High_Risk", knn_win, 1 - knn_win)
knn_cm <- confusionMatrix(knn_pred, y_test, positive = "High_Risk")
knn_cm
knn_roc <- roc(y_test, knn_prob, levels = c("Low_Risk", "High_Risk"), direction = "<")
auc(knn_roc)

# ---- SECTION 15: CONFUSION MATRIX COMPARISON ----
get_counts <- function(cm) {
  t <- cm$table
  c(TN = t["Low_Risk", "Low_Risk"], FP = t["High_Risk", "Low_Risk"],
    FN = t["Low_Risk", "High_Risk"], TP = t["High_Risk", "High_Risk"])
}
cm_compare <- rbind(KNN = get_counts(knn_cm), Decision_Tree = get_counts(tree_cm))
colnames(cm_compare) <- c("TN", "FP", "FN", "TP")
cm_compare

# ---- SECTION 16: MODEL COMPARISON TABLE AND CHART ----
get_metrics <- function(cm, roc_obj) {
  c(Accuracy = unname(cm$overall["Accuracy"]),
    Precision = unname(cm$byClass["Precision"]),
    Recall = unname(cm$byClass["Recall"]),
    Specificity = unname(cm$byClass["Specificity"]),
    F1 = unname(cm$byClass["F1"]),
    Balanced_Accuracy = unname(cm$byClass["Balanced Accuracy"]),
    ROC_AUC = as.numeric(auc(roc_obj)))
}
comparison <- rbind(KNN = get_metrics(knn_cm, knn_roc),
                    Decision_Tree = get_metrics(tree_cm, tree_roc))
round(comparison, 4)

plot_df <- data.frame(
  Model  = rep(rownames(comparison), times = ncol(comparison)),
  Metric = rep(colnames(comparison), each = nrow(comparison)),
  Value  = as.vector(comparison))
ggplot(plot_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("KNN" = "#2E86C1", "Decision_Tree" = "#C0392B")) +
  labs(title = "KNN vs Decision Tree - Performance Comparison", y = "Score") +
  theme_minimal() + theme(axis.text.x = element_text(angle = 30, hjust = 1))
