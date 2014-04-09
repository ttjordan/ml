library(matrixStats)
rm(list = ls())

sigmoid = function (z) {
  return (1 / (1 + exp(-z) ))
}

h = function (X, theta) {
  return(sigmoid(X %*% theta))
}

predictLabels <- function(X, W) {
  X %*% W
}

computeError <- function(Y, Ypredicted) {
  sum((Ypredicted - Y)^2) / nrow(Y)
}

computeCost <- function(X, Y, theta, m) {
  return(1/m * sum(-Y * log(h(X, theta)) - (1 - Y) * log(1 - h(X, theta))))
}

normalizeData <- function(X, mu, sd) {
  X_norm = X
  
  X_norm <- sweep(X_norm, 2, mu, "-")
  X_norm <- sweep(X_norm, 2, sd, "/")
  return(X_norm)
}

gradientDescent <- function(X, Y, theta, alpha, numberIterations, m) {
  for(x in 1:numberIterations) {
    theta <- theta - alpha / m * t(rowSums(X %*% theta - Y) %*% X)
    J_history[x] <<- computeCost(X, Y, theta, m)
  }
  return(theta)
}


numFolds <- 10
lockBinding("numFolds", globalenv())

numTrainingIter <- 50
lockBinding("numTrainingIter", globalenv())

spamData = read.csv("/Users/tjordan/Desktop/ML/Data/spambase/spambase.data", header=FALSE))

index <- 1:nrow(spamData)
index <- sample(index) ### shuffle index
fold <- rep(1:numFolds, each=nrow(spamData)/numFolds)[1:nrow(spamData)]

folds <- split(index, fold) ### create list with indices for each fold

minETrain = 999999999
minETest = 999999999
ETrain <- 1:numFolds
ETest <- 1:numFolds
bestTheta <- 1:ncol(spamData)

fullData <- rbind( data.matrix(spamData[, c(1:ncol(spamData) - 1)]))

for(x in 1:100) {
  for(i in 1:numFolds) {
    trainingSet <- spamData[-1 * unlist(folds[i]),]
    validationSet <- spamData[unlist(folds[i]),]
    
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
    
    numberOfIterations <- 60
    alpha <- 0.01
    theta <- data.matrix(rep(0,ncol(XTrainingMatrix_norm))) 
    J_history <- rep(0,numberOfIterations) 
    
    computeCost(XTrainingMatrix_norm, YTrainingMatrix, theta, length(YTrainingMatrix))
    
    theta <- gradientDescent(XTrainingMatrix_norm, YTrainingMatrix, theta, alpha, numberOfIterations, length(YTrainingMatrix))
    
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