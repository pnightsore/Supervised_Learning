#------------------------SUPERVISED LEARNING: BINARY CLASSIFICATION--------------------
library(ggplot2) # Visualize data elegantly.
library(dplyr) # Manipulate data efficiently.
library(corrplot)# Correlation matrix
library(reshape2) # Data reshaping.
library(caret) # Machine learning tools.
library(rpart) # Decision tree Model.
library(rpart.plot) # Decision tree plot.
library(class)  # KNN


#----------------1. Data Collection-------------------
# This dataset is collected from Kaggle
wine_data <- read.csv("/Users/macos/Desktop/DSE/Statistical learning/Supervised/Dataset/winequality-red.csv")

#----------------2. Data Preprocessing-----------------
# Summary statistics
summary(wine_data)

# Check for missing values
colSums(is.na(wine_data))

# View correlation matrix to identify important features
correlation_matrix <- cor(wine_data)
print(correlation_matrix)
# Convert 'quality' to a binary outcome (e.g., high quality vs. low quality)
# Assuming 'high quality' is defined as quality >= 6
wine_data$quality <- ifelse(wine_data$quality >= 6, "High", "Low")

# Convert to factor
wine_data$quality <- as.factor(wine_data$quality)

# Normalize or standardize the features (excluding the target variable)
scaled_data <- wine_data %>%
  select(-quality) %>%
  scale()  # Standardizes to have mean = 0 and standard deviation = 1

# Combine scaled features with the target variable
wine_data_scaled <- data.frame(scaled_data, quality = wine_data$quality)

# Histogram of all variables
melted_data <- melt(wine_data_scaled, id.vars = "quality")
p1 <- ggplot(melted_data, aes(value)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  facet_wrap(~ variable, scales = 'free') +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white")) +
  labs(title = "Histogram of All Variables", x = "Value", y = "Frequency")
ggsave("./plots/histogram_all_variables.png", plot = p1, bg = "white")

# Visualization of Wine Quality by Wine Type - Bar Chart
p2 <- ggplot(wine_data_scaled, aes(x = quality, fill = quality)) +  
  geom_bar(stat = "count") +  
  scale_fill_manual(values = c("Low" = "blue", "High" = "green")) +  
  theme_minimal() +  
  theme(plot.background = element_rect(fill = "white"),  
        panel.background = element_rect(fill = "white")) +  
  labs(title = "Wine Quality Distribution", x = "Wine Quality", y = "Count")  
ggsave("./plots/wine_quality_distribution.png", plot = p2, bg = "white")

# Box Plot
p3 <- ggplot(melted_data, aes(x = variable, y = value, fill = quality)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white")) +
  labs(title = "Box Plot of All Variables", x = "Variables", y = "Value")
ggsave("./plots/box_plot_all_variables.png", plot = p3, bg = "white")

# Scatter Plot
p4 <- ggplot(wine_data_scaled, aes(x = alcohol, y = pH, color = quality)) +
  geom_point(alpha = 0.7) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "white"),
        panel.background = element_rect(fill = "white")) +
  labs(title = "Scatter Plot of Alcohol vs pH", x = "Alcohol", y = "pH")
ggsave("./plots/scatter_plot_alcohol_vs_ph.png", plot = p4, bg = "white")

# Correlation Heatmap
png("./plots/correlation_heatmap.png", width = 800, height = 600, bg = "white")
corrplot(correlation_matrix, method = "color", type = "full",
         tl.col = "black", tl.srt = 45, 
         title = "Correlation Heatmap",
         mar = c(0,0,1,0))
dev.off()

#----------------3. Data Splitting-------------------
# Set seed for reproducibility
set.seed(123)

# Split data into training and testing sets
train_index <- createDataPartition(wine_data_scaled$quality, p = 0.7, list = FALSE)
train_data <- wine_data_scaled[train_index, ]
test_data <- wine_data_scaled[-train_index, ]


#----------------4. Model Selection-----------------

# Logistic Regression
logistic_model <- glm(quality ~ ., data = train_data, family = "binomial")
logistic_pred <- predict(logistic_model, newdata = test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, "High", "Low")

# Decision Tree
tree_model <- rpart(quality ~ ., data = train_data, method = "class")
#rpart.plot(tree_model)
tree_pred <- predict(tree_model, newdata = test_data, type = "class")



#----------------5. Hyperparameter Tuning-----------------
# Set up the range of k values to test  
k_values <- 1:20  
accuracy <- numeric(length(k_values))  

set.seed(123)  # For reproducibility  

# Cross-validation to find the best k  
for (k in k_values) {  
    # Predictions  
    knn_pred <- knn(train = train_data[,-ncol(train_data)],   
                    test = test_data[,-ncol(test_data)],   
                    cl = train_data$quality,   
                    k = k)  
    
    # Confusion matrix and accuracy  
    cm <- confusionMatrix(knn_pred, as.factor(test_data$quality))  
    accuracy[k] <- cm$overall['Accuracy']  
}  

# Save the plot as a PNG file  
png("./plots/knn_accuracy_plot.png", width = 800, height = 600)  

# Plot the accuracy against k values  
plot(k_values, accuracy, type = 'b', pch = 19,   
     xlab = "Number of Neighbors (k)",   
     ylab = "Accuracy",   
     main = "k-NN Accuracy vs. k Value",  
     col = "blue")  

# Highlight the best k  
best_k <- which.max(accuracy)  
points(best_k, accuracy[best_k], col = "red", pch = 19)  
text(best_k, accuracy[best_k], labels = paste("Best k:", best_k), pos = 3)  

# Close the device  
dev.off()

# k-Nearest Neighbors
k <- 5 # Optimal parameter
knn_pred <- knn(train = train_data[,-ncol(train_data)], test = test_data[,-ncol(test_data)], cl = train_data$quality, k = k)



#----------------6. Model Evaluation-----------------

# Logistic Regression Evaluation
logistic_conf <- confusionMatrix(as.factor(logistic_pred_class), as.factor(test_data$quality))
print("Logistic Regression:")
print(logistic_conf)
logistic_precision <- posPredValue(logistic_conf$table)
logistic_recall <- sensitivity(logistic_conf$table)
logistic_f1 <- 2 * logistic_precision * logistic_recall / (logistic_precision + logistic_recall)
print(paste("Precision: ", logistic_precision))
print(paste("Recall: ", logistic_recall))
print(paste("F1 Score: ", logistic_f1))

# Decision Tree Evaluation
tree_conf <- confusionMatrix(as.factor(tree_pred), as.factor(test_data$quality))
print("Decision Tree:")
print(tree_conf)
tree_precision <- posPredValue(tree_conf$table)
tree_recall <- sensitivity(tree_conf$table)
tree_f1 <- 2 * tree_precision * tree_recall / (tree_precision + tree_recall)
print(paste("Precision: ", tree_precision))
print(paste("Recall: ", tree_recall))
print(paste("F1 Score: ", tree_f1))

# k-Nearest Neighbors Evaluation
knn_conf <- confusionMatrix(as.factor(knn_pred), as.factor(test_data$quality))
print("k-Nearest Neighbors:")
print(knn_conf)
knn_precision <- posPredValue(knn_conf$table)
knn_recall <- sensitivity(knn_conf$table)
knn_f1 <- 2 * knn_precision * knn_recall / (knn_precision + knn_recall)
print(paste("Precision: ", knn_precision))
print(paste("Recall: ", knn_recall))
print(paste("F1 Score: ", knn_f1))


#----------------7. Model Visualization-----------------
# Decision Tree Visualization
png("./plots/decision_tree.png", width = 800, height = 600, bg = "white")
rpart.plot(tree_model, main = "Decision Tree for Wine Quality")
dev.off()
