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
  
  
  meanData <- colMeans(trainingSet)
  for(j in 1:ncol(trainingSetPositive_dataMatrix)) {
    trainingSetPositive_dataMatrix[,j] <- as.numeric( trainingSetPositive_dataMatrix[,j] > meanData[j] ) 
    trainingSetNegative_dataMatrix[,j] <- as.numeric( trainingSetNegative_dataMatrix[,j] > meanData[j] ) 
    validationSetPositive_dataMatrix[,j] <- as.numeric( validationSetPositive_dataMatrix[,j] > meanData[j] ) 
    validationSetNegative_dataMatrix[,j] <- as.numeric( validationSetNegative_dataMatrix[,j] > meanData[j] ) 
  }

  
  probabilityPositive <- nrow(trainingSetPositive_dataMatrix) / nrow(trainingSet)
  probabilityNegative <- nrow(trainingSetNegative_dataMatrix) / nrow(trainingSet)
  
  positiveSpamProbabilitiesPerFeature <- mat.or.vec(1, ncol(trainingSetPositive_dataMatrix))
  for(x in 1:nrow(trainingSetPositive_dataMatrix)) {
    positiveSpamProbabilitiesPerFeature <- positiveSpamProbabilitiesPerFeature + 1 * (trainingSetPositive_dataMatrix[x,] == 1)
  }
  positiveSpamProbabilitiesPerFeature <- positiveSpamProbabilitiesPerFeature + 1
  positiveSpamProbabilitiesPerFeature <- positiveSpamProbabilitiesPerFeature / (2 + nrow(trainingSetPositive_dataMatrix)) 
  
  negativeNonSpamProbabilitiesPerFeature <- mat.or.vec(1, ncol(trainingSetNegative_dataMatrix))
  for(x in 1:nrow(trainingSetNegative_dataMatrix)) {
    negativeNonSpamProbabilitiesPerFeature <- negativeNonSpamProbabilitiesPerFeature + 1 * (trainingSetNegative_dataMatrix[x,] == 1)
  }
  negativeNonSpamProbabilitiesPerFeature <- negativeNonSpamProbabilitiesPerFeature + 1
  negativeNonSpamProbabilitiesPerFeature <- negativeNonSpamProbabilitiesPerFeature / (2 + nrow(trainingSetNegative_dataMatrix)) 
  
  trainingSetPositive_dataMatrixNeg <-  1 - trainingSetPositive_dataMatrix
  trainingSetNegative_dataMatrixNeg <-  1 - trainingSetNegative_dataMatrix
  validationSetPositive_dataMatrixNeg <-  1 - validationSetPositive_dataMatrix
  validationSetNegative_dataMatrixNeg <-  1 - validationSetNegative_dataMatrix
  #TEST on TRAINING
  predictedPositiveProbabilityTemp <- sweep(trainingSetPositive_dataMatrix ,MARGIN=2,positiveSpamProbabilitiesPerFeature,'*') 
  predictedPositiveProbabilityTempNeg <- sweep(trainingSetPositive_dataMatrixNeg ,MARGIN=2, 1 - positiveSpamProbabilitiesPerFeature,'*') 
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(predictedPositiveProbabilityTemp), 1))
  for(y in 1:nrow(predictedPositiveProbabilityTemp)) {
    temp <- predictedPositiveProbabilityTemp[y,]
    tempNeg <- predictedPositiveProbabilityTempNeg[y,]
    predictedPositiveProbability[y] <- sum(log(temp[temp != 0])) + sum(log(tempNeg[tempNeg != 0]))
  }
  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  predictedNegativeProbabilityTemp <- sweep(trainingSetPositive_dataMatrix ,MARGIN=2,negativeNonSpamProbabilitiesPerFeature,'*') 
  predictedNegativeProbabilityTempNeg <- sweep(trainingSetPositive_dataMatrixNeg ,MARGIN=2, 1 - negativeNonSpamProbabilitiesPerFeature,'*') 
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(predictedNegativeProbabilityTemp), 1))
  for(y in 1:nrow(predictedNegativeProbabilityTemp)) {
    temp <- predictedNegativeProbabilityTemp[y,]
    tempNeg <- predictedNegativeProbabilityTempNeg[y,]
    predictedNegativeProbability[y] <- sum(log(temp[temp != 0]))+ sum(log(tempNeg[tempNeg != 0]))
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  
  TP <- sum(predictedPositiveProbabilityFinal > predictedNegativeProbabilityFinal)
  FN <-  nrow(predictedPositiveProbabilityFinal) - TP
  
  predictedPositiveProbabilityTemp <- sweep(trainingSetNegative_dataMatrix ,MARGIN=2,positiveSpamProbabilitiesPerFeature,'*') 
  predictedPositiveProbabilityTempNeg <- sweep(trainingSetNegative_dataMatrixNeg ,MARGIN=2, 1 - positiveSpamProbabilitiesPerFeature,'*') 
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(predictedPositiveProbabilityTemp), 1))
  for(y in 1:nrow(predictedPositiveProbabilityTemp)) {
    temp <- predictedPositiveProbabilityTemp[y,]
    tempNeg <- predictedPositiveProbabilityTempNeg[y,]
    predictedPositiveProbability[y] <- sum(log(temp[temp != 0]))+ sum(log(tempNeg[tempNeg != 0])) 
  }
  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  predictedNegativeProbabilityTemp <- sweep(trainingSetNegative_dataMatrix ,MARGIN=2,negativeNonSpamProbabilitiesPerFeature,'*') 
  predictedNegativeProbabilityTempNeg <- sweep(trainingSetNegative_dataMatrixNeg ,MARGIN=2, 1 - negativeNonSpamProbabilitiesPerFeature,'*') 
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(predictedNegativeProbabilityTemp), 1))
  for(y in 1:nrow(predictedNegativeProbabilityTemp)) {
    temp <- predictedNegativeProbabilityTemp[y,]
    tempNeg <- predictedNegativeProbabilityTempNeg[y,]
    predictedNegativeProbability[y] <- sum(log(temp[temp != 0]))+ sum(log(tempNeg[tempNeg != 0]))
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  TN <- sum(predictedPositiveProbabilityFinal < predictedNegativeProbabilityFinal)
  FP <- nrow(predictedPositiveProbabilityFinal) - TN
  
  trainAccuracy <- trainAccuracy +  (TP + TN)/(TP+TN+FP+FN)
  
  ##### Test set
  predictedPositiveProbabilityTemp <- sweep(validationSetPositive_dataMatrix ,MARGIN=2,positiveSpamProbabilitiesPerFeature,'*') 
  predictedPositiveProbabilityTempNeg <- sweep(validationSetPositive_dataMatrixNeg ,MARGIN=2, 1 - positiveSpamProbabilitiesPerFeature,'*') 
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(predictedPositiveProbabilityTemp), 1))
  for(y in 1:nrow(predictedPositiveProbabilityTemp)) {
    temp <- predictedPositiveProbabilityTemp[y,]
    tempNeg <- predictedPositiveProbabilityTempNeg[y,]
    predictedPositiveProbability[y] <- sum(log(temp[temp != 0])) + sum(log(tempNeg[tempNeg != 0]))
  }
  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  predictedNegativeProbabilityTemp <- sweep(validationSetPositive_dataMatrix ,MARGIN=2,negativeNonSpamProbabilitiesPerFeature,'*') 
  predictedNegativeProbabilityTempNeg <- sweep(validationSetPositive_dataMatrixNeg ,MARGIN=2, 1 - negativeNonSpamProbabilitiesPerFeature,'*') 
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(predictedNegativeProbabilityTemp), 1))
  for(y in 1:nrow(predictedNegativeProbabilityTemp)) {
    temp <- predictedNegativeProbabilityTemp[y,]
    tempNeg <- predictedNegativeProbabilityTempNeg[y,]
    predictedNegativeProbability[y] <- sum(log(temp[temp != 0]))+ sum(log(tempNeg[tempNeg != 0]))
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  
  TP <- sum(predictedPositiveProbabilityFinal > predictedNegativeProbabilityFinal)
  FN <-  nrow(predictedPositiveProbabilityFinal) - TP
  
  predictedPositiveProbabilityTemp <- sweep(validationSetNegative_dataMatrix ,MARGIN=2,positiveSpamProbabilitiesPerFeature,'*') 
  predictedPositiveProbabilityTempNeg <- sweep(validationSetNegative_dataMatrixNeg ,MARGIN=2, 1 - positiveSpamProbabilitiesPerFeature,'*') 
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(predictedPositiveProbabilityTemp), 1))
  for(y in 1:nrow(predictedPositiveProbabilityTemp)) {
    temp <- predictedPositiveProbabilityTemp[y,]
    tempNeg <- predictedPositiveProbabilityTempNeg[y,]
    predictedPositiveProbability[y] <- sum(log(temp[temp != 0]))+ sum(log(tempNeg[tempNeg != 0]))
  }
  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  predictedNegativeProbabilityTemp <- sweep(validationSetNegative_dataMatrix ,MARGIN=2,negativeNonSpamProbabilitiesPerFeature,'*') 
  predictedNegativeProbabilityTempNeg <- sweep(validationSetNegative_dataMatrixNeg ,MARGIN=2, 1 - negativeNonSpamProbabilitiesPerFeature,'*') 
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(predictedNegativeProbabilityTemp), 1))
  for(y in 1:nrow(predictedNegativeProbabilityTemp)) {
    temp <- predictedNegativeProbabilityTemp[y,]
    tempNeg <- predictedNegativeProbabilityTempNeg[y,]
    predictedNegativeProbability[y] <- sum(log(temp[temp != 0])) + sum(log(tempNeg[tempNeg != 0]))
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  TN <- sum(predictedPositiveProbabilityFinal < predictedNegativeProbabilityFinal)
  FP <- nrow(predictedPositiveProbabilityFinal) - TN

  testAccuracy <- testAccuracy +  (TP + TN)/(TP+TN+FP+FN)
}

trainAccuracy <- trainAccuracy / numFolds
testAccuracy <- testAccuracy / numFolds

print("Average train error:")
print(1-trainAccuracy)
print("Average test error:")
print(1-testAccuracy)