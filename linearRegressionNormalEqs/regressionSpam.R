rm(list=ls())

train <- function(X, Y) {
  solve(t(X) %*% X) %*% t(X) %*% Y
}

predictLabels <- function(X, W) {
  X %*% W
}

computeError <- function(Y, Ypredicted) {
  sum((Ypredicted - Y)^2) / nrow(Y)
}

numFolds <- 10
lockBinding("numFolds", globalenv())

numFeatures <- 57
lockBinding("numFeatures", globalenv())

trainingData <- data.matrix(read.csv("/Users/tjordan/Desktop/ML/Data/spambase/spambase.data", header=FALSE)))

index <- 1:nrow(trainingData)
index <- sample(index) ### shuffle index
fold <- rep(1:numFolds, each=nrow(trainingData)/numFolds)[1:nrow(trainingData)]

folds <- split(index, fold) ### create list with indices for each fold

minETrain = 999999999
minETest = 999999999
ETrain <- 1:numFolds
ETest <- 1:numFolds
bestWeights <- 1:ncol(trainingData)

for(x in 1:100) {
  for(i in 1:numFolds) {
    trainingSet <- trainingData[-1 * unlist(folds[i]),]
    validationSet <- trainingData[unlist(folds[i]),]
    
    XTrain <- trainingSet[ ,1:numFeatures]
    XTest <- validationSet[ ,1:numFeatures]
    
    YTrain <- data.matrix(trainingSet[ ,numFeatures + 1])
    YTest <- data.matrix(validationSet[ ,numFeatures + 1])
    
    W <- train(XTrain,YTrain)
    
    PredictedYTrain <- predictLabels(XTrain, W)
    PredictedYTest <- predictLabels(XTest, W)
    
    ETrain[i] <- computeError(YTrain, PredictedYTrain)
    ETest[i] <- computeError(YTest, PredictedYTest)
    
    if(minETrain > ETrain[i]) {
      minETrain <- ETrain[i]
    }
    
    if(minETest > ETest[i]) {
      minETest <- ETest[i]
      bestWeights <- W
    }
  }
}
