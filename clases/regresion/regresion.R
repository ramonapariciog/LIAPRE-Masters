# 0. Build linear model 
data("cars", package="datasets")
model <- lm (dist ~ speed, data=cars)
# 1. Add predictions 
new.speeds <- data.frame(speed=c(12, 19, 24))
# predicted <- predict(model, newdata=new.speeds, interval="confidence")
pred.int <- predict(model, interval="prediction")
mydata <- cbind(cars, pred.int)
# 2. Regression line + confidence intervals
library("ggplot2")
p <- ggplot(mydata, aes(speed, dist)) +
  geom_point() +
  stat_smooth(method = lm)
# 3. Add prediction intervals
p + geom_line(aes(y = lwr), color = "red", linetype = "dashed")+
  geom_line(aes(y = upr), color = "red", linetype = "dashed")
