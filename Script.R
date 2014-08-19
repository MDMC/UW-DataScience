setwd("~/Documents/Titanic")
library(caret)

#Read the data into R
data = read.csv("train.csv", sep=",", header=TRUE)
pairs(data)

#Set aside some of the data to use for cross validation
data$Survived=as.factor(data$Survived)
set.seed(123)
inTrain <- createDataPartition(y=data$Survived, p=.7, list=FALSE)
training <- data[inTrain,]
testing <-data[-inTrain,]

# Build a Random Forest using all of the available variables
modFit <- train(Survived ~ ., data=training, method="rf", prox=TRUE)
modFit
varImp(modFit)
#The four most important variables were gender, fare, Pclass, and Age.
# Rerun the algorithm with just those variables, and see what happens to the accuracy
modFit <- train(Survived ~ Sex + Pclass + Fare + Age, data=training, method="rf", prox=TRUE)
modFit
varImp(modFit)
#The Fare variable is a bit compounded by instances where the fare is apparently for multiple
# passengers.  Further, dropping the Fare variable results in only a small decrease in accuracy
# (.006)  For simplicity, I'll remove this from the model.

# A significant proportion of the Age information is NA.
# I chose a simple imputation method of replacing the missing ages
# with the age average of the training data.
mean=29.78
training$Age[which(is.na(training$Age))] <- mean
testing$Age[which(is.na(test$Age))] <- mean
modFit <- train(Survived ~ Sex + Pclass + Age, data=training, method="rf", prox=TRUE)
testPred = predict(modFit, testing)

confusionMatrix(testPred, testing$Survived)

#More work could be done to mine the fields for more information.  For example,
# young men seem to be listed as "Master", so some age information could be imputed
# where there are na values for Age.
# Also, some work could possibly be done to split the multi-person Fares across the
# passengers (possibly by last name), making this variable potentially more useful.
#
# OK:  Lets evaluate our results!
data = read.csv("test.csv", sep=",", header=TRUE)
data$Age[which(is.na(data$Age))] <- mean
testPred=predict(modFit, data)

results <- as.data.frame(cbind(data$PassengerId, testPred)
results$testPred <- as.factor(results$testPred)
results$testPred <- factor(results$testPred, labels=c("0", "1"))
names(results) <- c("PassengerId", "Survived")
write.csv(results, file="model.csv", row.names=FALSE)
