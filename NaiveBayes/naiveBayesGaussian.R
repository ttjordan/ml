rm(list=ls())
library(matrixStats)


numFolds <- 10
lockBinding("numFolds", globalenv())

spamData <- read.csv("/Users/tjordan/Desktop/ML/Data/spambase/spambase.data", header=FALSE)

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
  
  lambda <- ncol(trainingSetPositive_dataMatrix)/(2 + ncol(trainingSetPositive_dataMatrix))
  backgroundSD <- colSds(rbind(trainingSetPositive_dataMatrix,trainingSetNegative_dataMatrix)) 
  foregroundPositiveSD <- colSds(trainingSetPositive_dataMatrix)
  foregroundNegativeSD <- colSds(trainingSetNegative_dataMatrix)
    
  meanPositiveData <- colMeans(trainingSetPositive_dataMatrix)
  sdPositiveData <- lambda * foregroundPositiveSD + (1 - lambda) * backgroundSD * 0.5
  meanNegativeData <- colMeans(trainingSetNegative_dataMatrix)
  sdNegativeData <- lambda * foregroundNegativeSD + (1 - lambda) * backgroundSD * 0.5
  
  probabilityPositive <- nrow(trainingSetPositive_dataMatrix) / nrow(trainingSet)
  probabilityNegative <- nrow(trainingSetNegative_dataMatrix) / nrow(trainingSet)

  #TEST on TRAINING
  positiveProbabilities <- mat.or.vec(nrow(trainingSetPositive_dataMatrix), ncol(trainingSetPositive_dataMatrix))
  for(column in 1:ncol(trainingSetPositive_dataMatrix)) {
    positiveProbabilities[,column] <- dnorm(trainingSetPositive_dataMatrix[, column], mean = meanPositiveData[column], sd = sdPositiveData[column], log = TRUE)
  }
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(positiveProbabilities), 1))
  
  for(row in 1:nrow(positiveProbabilities)) {
    predictedPositiveProbability[row] <- sum(positiveProbabilities[row,])
  }

  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  negativeProbabilities <-  mat.or.vec(nrow(trainingSetPositive_dataMatrix), ncol(trainingSetPositive_dataMatrix))
  for(column in 1:ncol(trainingSetPositive_dataMatrix)) {
    negativeProbabilities[,column] <- dnorm(trainingSetPositive_dataMatrix[, column], mean = meanNegativeData[column], sd = sdNegativeData[column], log = TRUE)
  }
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(negativeProbabilities), 1))
  
  for(row in 1:nrow(negativeProbabilities)) {
    predictedNegativeProbability[row] <- sum(negativeProbabilities[row,])
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  
  TP <- sum(predictedPositiveProbabilityFinal > predictedNegativeProbabilityFinal)
  FN <-  nrow(predictedPositiveProbabilityFinal) - TP
  
  positiveProbabilities <-  mat.or.vec(nrow(trainingSetNegative_dataMatrix), ncol(trainingSetNegative_dataMatrix))
  for(column in 1:ncol(trainingSetNegative_dataMatrix)) {
    positiveProbabilities[,column] <- dnorm(trainingSetNegative_dataMatrix[, column], mean = meanPositiveData[column], sd = sdPositiveData[column], log = TRUE)
  }
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(positiveProbabilities), 1))
  
  for(row in 1:nrow(positiveProbabilities)) {
    predictedPositiveProbability[row] <- sum(positiveProbabilities[row,])
  }
  
  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  negativeProbabilities <-  mat.or.vec(nrow(trainingSetNegative_dataMatrix), ncol(trainingSetNegative_dataMatrix))
  for(column in 1:ncol(trainingSetNegative_dataMatrix)) {
    negativeProbabilities[,column] <- dnorm(trainingSetNegative_dataMatrix[, column], mean = meanNegativeData[column], sd = sdNegativeData[column], log = TRUE)
  }
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(negativeProbabilities), 1))
  
  for(row in 1:nrow(negativeProbabilities)) {
    predictedNegativeProbability[row] <- sum(negativeProbabilities[row,])
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  

  TN <- sum(predictedPositiveProbabilityFinal < predictedNegativeProbabilityFinal)
  FP <- nrow(predictedPositiveProbabilityFinal) - TN
  
  trainAccuracy <- trainAccuracy +  (TP + TN)/(TP+TN+FP+FN)
  print("training")
  print((TP + TN)/(TP+TN+FP+FN))
  ##### Test set
  positiveProbabilities <- mat.or.vec(nrow(validationSetPositive_dataMatrix), ncol(validationSetPositive_dataMatrix))
  for(column in 1:ncol(validationSetPositive_dataMatrix)) {
    positiveProbabilities[,column] <- dnorm(validationSetPositive_dataMatrix[, column], mean = meanPositiveData[column], sd = sdPositiveData[column], log = TRUE)
  }
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(positiveProbabilities), 1))
  
  for(row in 1:nrow(positiveProbabilities)) {
    predictedPositiveProbability[row] <- sum(positiveProbabilities[row,])
  }
  
  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  negativeProbabilities <-  mat.or.vec(nrow(validationSetPositive_dataMatrix), ncol(validationSetPositive_dataMatrix))
  for(column in 1:ncol(trainingSetPositive_dataMatrix)) {
    negativeProbabilities[,column] <- dnorm(validationSetPositive_dataMatrix[, column], mean = meanNegativeData[column], sd = sdNegativeData[column], log = TRUE)
  }
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(negativeProbabilities), 1))
  
  for(row in 1:nrow(negativeProbabilities)) {
    predictedNegativeProbability[row] <- sum(negativeProbabilities[row,])
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  
  TP <- sum(predictedPositiveProbabilityFinal > predictedNegativeProbabilityFinal)
  FN <-  nrow(predictedPositiveProbabilityFinal) - TP
  
  positiveProbabilities <-  mat.or.vec(nrow(validationSetNegative_dataMatrix), ncol(validationSetNegative_dataMatrix))
  for(column in 1:ncol(trainingSetNegative_dataMatrix)) {
    positiveProbabilities[,column] <- dnorm(validationSetNegative_dataMatrix[, column], mean = meanPositiveData[column], sd = sdPositiveData[column], log = TRUE)
  }
  
  predictedPositiveProbability <- data.matrix(mat.or.vec(nrow(positiveProbabilities), 1))
  
  for(row in 1:nrow(positiveProbabilities)) {
    predictedPositiveProbability[row] <- sum(positiveProbabilities[row,])
  }
  
  predictedPositiveProbabilityFinal <- predictedPositiveProbability + log(probabilityPositive)
  
  negativeProbabilities <-  mat.or.vec(nrow(validationSetNegative_dataMatrix), ncol(validationSetNegative_dataMatrix))
  for(column in 1:ncol(validationSetNegative_dataMatrix)) {
    negativeProbabilities[,column] <- dnorm(validationSetNegative_dataMatrix[, column], mean = meanNegativeData[column], sd = sdNegativeData[column], log = TRUE)
  }
  
  predictedNegativeProbability <- data.matrix(mat.or.vec(nrow(negativeProbabilities), 1))
  
  for(row in 1:nrow(negativeProbabilities)) {
    predictedNegativeProbability[row] <- sum(negativeProbabilities[row,])
  }
  predictedNegativeProbabilityFinal <- predictedNegativeProbability + log(probabilityNegative)
  
  
  
  TN <- sum(predictedPositiveProbabilityFinal < predictedNegativeProbabilityFinal)
  FP <- nrow(predictedPositiveProbabilityFinal) - TN
  
  testAccuracy <- testAccuracy +  (TP + TN)/(TP+TN+FP+FN)
  print("validation")
  print((TP + TN)/(TP+TN+FP+FN))
  
}

trainAccuracy <- trainAccuracy / numFolds
testAccuracy <- testAccuracy / numFolds

print("Average train error:")
print(1-trainAccuracy)
print("Average test error:")
print(1-testAccuracy)