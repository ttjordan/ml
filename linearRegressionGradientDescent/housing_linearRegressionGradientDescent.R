rm(list = ls())
library(matrixStats)

computeCost <- function(X, Y, theta) {
  J <- 0.0
  m <- 2 * length(Y)
  J <- sum((X %*% theta - Y) * (X %*% theta - Y)) / m
  return(J)
}

predictLabels <- function(X, W) {
  X %*% W
}

computeError <- function(Y, Ypredicted) {
  sum((Ypredicted - Y)^2) / nrow(Y)
}

normalizeData <- function(X, mu, sd) {
  X_norm = X
  
  X_norm <- sweep(X_norm, 2, mu, "-")
  X_norm <- sweep(X_norm, 2, sd, "/")
  return(X_norm)
}

gradientDescent <- function(X, Y, theta, alpha, numberIterations) {
  m = length(Y)
  for(x in 1:numberIterations) {
    x
    theta <- theta - alpha / m * t(rowSums(X %*% theta - Y) %*% X)
    J_history[x] <<- computeCost(X, Y, theta)
  }
  return(theta)
}

numFolds <- 10
lockBinding("numFolds", globalenv())

numTrainingIter <- 50
lockBinding("numTrainingIter", globalenv())

housingData = read.table("/Users/tjordan/Desktop/ML/Data/housing_train.txt")
testData <- data.matrix(read.table("/Users/tjordan/Desktop/ML/Data/housing_test.txt"))

index <- 1:nrow(housingData)
index <- sample(index) ### shuffle index
fold <- rep(1:numFolds, each=nrow(housingData)/numFolds)[1:nrow(housingData)]

folds <- split(index, fold) ### create list with indices for each fold

minETrain = 999999999
minETest = 999999999
ETrain <- 1:numFolds
ETest <- 1:numFolds
bestTheta <- 1:ncol(housingData)

X_TestData <- testData[, c(1:ncol(testData) - 1)]
XTestMatrix <- data.matrix(X_TestData)
fullData <- rbind( data.matrix(housingData[, c(1:ncol(housingData) - 1)]),  X_TestData)

for(x in 1:numTrainingIter) {
  for(i in 1:numFolds) {
    trainingSet <- housingData[-1 * unlist(folds[i]),]
    validationSet <- housingData[unlist(folds[i]),]
    
    XTraining_data <- trainingSet[, c(1:ncol(trainingSet) - 1)]
    YTraining_data <- trainingSet[, c(ncol(trainingSet))]
    
    XTrainingTest_data <- validationSet[, c(1:ncol(validationSet) - 1)]
    YTrainingTest_data <- validationSet[, c(ncol(validationSet))]
    
    XTrainingMatrix <- data.matrix(XTraining_data)
    YTrainingMatrix <- data.matrix(YTraining_data)
    
    XTrainingTestMatrix <- data.matrix(XTrainingTest_data)
    YTrainingTestMatrix <- data.matrix(YTrainingTest_data)

    XTrainingMatrix_norm = normalizeData(XTrainingMatrix, colMeans(fullData), colSds(fullData))
    XTrainingTestMatrix_norm = normalizeData(XTrainingTestMatrix, colMeans(fullData), colSds(fullData))
    
    XTrainingMatrix_norm <- cbind(rep(1,nrow(XTrainingMatrix_norm)), XTrainingMatrix_norm)
    XTrainingTestMatrix_norm <- cbind(rep(1,nrow(XTrainingTestMatrix_norm)), XTrainingTestMatrix_norm)
    
    numberOfIterations <- 20
    alpha <- 0.1
    theta <- data.matrix(rep(0,ncol(XTrainingMatrix_norm))) 
    J_history <- rep(0,numberOfIterations) 
    
    computeCost(XTrainingMatrix_norm, YTrainingMatrix, theta)
    
    theta <- gradientDescent(XTrainingMatrix_norm, YTrainingMatrix, theta, alpha, numberOfIterations)
    
    PredictedYTrain <- predictLabels(XTrainingMatrix_norm, theta)
    PredictedYTest <- predictLabels(XTrainingTestMatrix_norm, theta)
    
    ETrain[i] <- computeError(YTrainingMatrix, PredictedYTrain)
    ETest[i] <- computeError(YTrainingTestMatrix, PredictedYTest)
    
    if(minETrain > ETrain[i]) {
      minETrain <- ETrain[i]
    }
    
    if(minETest > ETest[i]) {
      minETest <- ETest[i]
      bestTheta <- theta
      plot(J_history)
    }
  }
}


XTestMatrix_norm = normalizeData(XTestMatrix, colMeans(fullData), colSds(fullData))
XTestMatrix_norm <- cbind(rep(1,nrow(XTestMatrix_norm)), XTestMatrix_norm)

Y_TestData <- testData[, c(ncol(testData))]
YTestMatrix <- data.matrix(Y_TestData)

PredictedYTest <- predictLabels(XTestMatrix_norm, bestTheta)
TestError <- computeError(YTestMatrix, PredictedYTest)
