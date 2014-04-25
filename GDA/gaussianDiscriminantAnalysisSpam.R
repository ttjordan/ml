rm(list=ls())
library(matrixStats)
library(mnormt)

numFolds <- 10
lockBinding("numFolds", globalenv())

spamData = read.csv("/Users/tjordan/Desktop/ML/Data/spambase/spambase.data", header=FALSE)

index <- 1:nrow(spamData)
index <- sample(index) ### shuffle index
fold <- rep(1:numFolds, each=nrow(spamData)/numFolds)[1:nrow(spamData)]

folds <- split(index, fold) ### create list with indices for each fold
trainAccuracy <- 0
testAccuracy <- 0

for(i in 1:numFolds) {
  trainingSet <- spamData[-1 * unlist(folds[i]),]
  validationSet <- spamData[unlist(folds[i]),]
  
  trainingSetPositive <- subset(trainingSet, V58 == 1)
  trainingSetNegative <- subset(trainingSet, V58 == 0)
  validationSetPositive <- subset(validationSet, V58 == 1)
  validationSetNegative <- subset(validationSet, V58 == 0)
  
  trainingSetPositive_data <- trainingSetPositive[, c(1:ncol(trainingSetPositive) - 1)]
  trainingSetPositive_labels <- trainingSetPositive[, c(ncol(trainingSetPositive))]
  validationSetPositive_data <- validationSetPositive[, c(1:ncol(validationSetPositive) - 1)]
  validationSetPositive_labels <- validationSetPositive[, c(ncol(validationSetPositive))]
  
  trainingSetNegative_data <- trainingSetNegative[, c(1:ncol(trainingSetNegative) - 1)]
  trainingSetNegative_labels <- trainingSetNegative[, c(ncol(trainingSetNegative))]
  validationSetNegative_data <- validationSetNegative[, c(1:ncol(validationSetNegative) - 1)]
  validationSetNegative_labels <- validationSetNegative[, c(ncol(validationSetNegative))]
  
  trainingSetPositive_dataMatrix <- data.matrix(trainingSetPositive_data)
  trainingSetPositive_labelsMatrix <- data.matrix(trainingSetPositive_labels)
  validationSetPositive_dataMatrix <- data.matrix(validationSetPositive_data)
  validationSetPositive_labelsMatrix <- data.matrix(validationSetPositive_labels)
  
  trainingSetNegative_dataMatrix <- data.matrix(trainingSetNegative_data)
  trainingSetNegative_labelsMatrix <- data.matrix(trainingSetNegative_labels)
  validationSetNegative_dataMatrix <- data.matrix(validationSetNegative_data)
  validationSetNegative_labelsMatrix <- data.matrix(validationSetNegative_labels)
  
  
  meanTrainingSetPositive_data <- colMeans(trainingSetPositive_dataMatrix)
  meanTrainingSetNegative_data <- colMeans(trainingSetNegative_dataMatrix)
  
  covariance <- cov(rbind(trainingSetPositive_dataMatrix, trainingSetNegative_dataMatrix))
  
  probabilityPositive <- nrow(trainingSetPositive_dataMatrix) / nrow(trainingSet)
  probabilityNegative <- nrow(trainingSetNegative_dataMatrix) / nrow(trainingSet)
  
  predictedPositiveProbability <- log(probabilityPositive) + data.matrix(dmnorm(trainingSetPositive_dataMatrix, meanTrainingSetPositive_data, covariance, log = TRUE))
  predictedNegativeProbability <- log(probabilityNegative) + data.matrix(dmnorm(trainingSetPositive_dataMatrix, meanTrainingSetNegative_data, covariance, log = TRUE))
  
  TP <- sum(predictedPositiveProbability > predictedNegativeProbability)
  FN <-  nrow(predictedPositiveProbability) - TP
  
  predictedPositiveProbability <- log(probabilityPositive) + data.matrix(dmnorm(trainingSetNegative_dataMatrix, meanTrainingSetPositive_data, covariance, log = TRUE))
  predictedNegativeProbability <- log(probabilityNegative) + data.matrix(dmnorm(trainingSetNegative_dataMatrix, meanTrainingSetNegative_data, covariance, log = TRUE))
  
  TN <- sum(predictedPositiveProbability < predictedNegativeProbability)
  FP <- nrow(predictedPositiveProbability) - TN
  
  trainAccuracy <- trainAccuracy +  (TP + TN)/(TP+TN+FP+FN)
  
 
  predictedPositiveProbability <- log(probabilityPositive) + data.matrix(dmnorm(validationSetPositive_dataMatrix, meanTrainingSetPositive_data, covariance, log = TRUE))
  predictedNegativeProbability <- log(probabilityNegative) + data.matrix(dmnorm(validationSetPositive_dataMatrix, meanTrainingSetNegative_data, covariance, log = TRUE))
  
  TP <- sum(predictedPositiveProbability > predictedNegativeProbability)
  FN <-  nrow(predictedPositiveProbability) - TP
  
  predictedPositiveProbability <- log(probabilityPositive) + data.matrix(dmnorm(validationSetNegative_dataMatrix, meanTrainingSetPositive_data, covariance, log = TRUE))
  predictedNegativeProbability <- log(probabilityNegative) + data.matrix(dmnorm(validationSetNegative_dataMatrix, meanTrainingSetNegative_data, covariance, log = TRUE))
  
  TN <- sum(predictedPositiveProbability < predictedNegativeProbability)
  FP <- nrow(predictedPositiveProbability) - TN
  
  testAccuracy <- testAccuracy +  (TP + TN)/(TP+TN+FP+FN)
}

trainAccuracy <- trainAccuracy / numFolds
testAccuracy <- testAccuracy / numFolds

