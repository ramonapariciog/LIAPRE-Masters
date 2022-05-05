library(MASS)
library(ggplot2)
library(dplyr)
attach(iris)
View(iris)
str(iris)
#scale each predictor variable (i.e. first 4 columns)
iris[,1:4] <- scale(iris[,1:4])
means <- apply(iris[,1:4], 2, mean)
stds <- apply(iris[,1:4], 2, sd)

#make this example reproducible
set.seed(1)

#Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE), nrow(iris), replace=TRUE,
                 prob=c(0.7,0.3))
train <- iris[sample, ]
test <- iris[!sample, ] 
# get the prior probabilities
prior <- 1 / dim(train)[1]
table(train["Species"]) * prior
prior <- table(train["Species"]) * prior
# model <- lda(Species~., data=train)
model <- lda(Species ~ ., data=train)
predicted <- model %>% predict(test)
trainres <- predict(model)
print(names(predicted))
accuracy_val <- mean(predicted$class==test$Species)
lda_plot <- cbind(train, trainres$x, trainres$class)
colnames(lda_plot)[colnames(lda_plot) %in% c("trainres$class")] <- c("PredictedClass")
#create plot
ggplot(lda_plot, aes(LD1, LD2)) +
  geom_point(aes(color = PredictedClass))
