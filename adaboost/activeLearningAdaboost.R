rm(list=ls())

voteData <- read.csv("/Users/tjordan/Desktop/ML/Data/vote/vote.txt", header=FALSE)

NUMITERATIONS <- 50

buildStump <- function(voteDataT, voteLabelsT, D) {
  numCols <- ncol(voteDataT)
  
  minStumpError <- Inf
  minErrorStump<-matrix(1,5)
  for(col in 1:numCols) {
    tempCol <-  data.matrix(voteDataT[,col])
    
    errorGreater <- Inf
    minErrorGreaterThreshold <- Inf
    yPredictGreater <- mat.or.vec(nrow(voteDataT), 1)
    #greater
    for(row in 1:nrow(voteDataT)) {
      index <- tempCol >= tempCol[row]
      yPredictGreater[index] <- 1
      yPredictGreater[!index] <- -1
      
      index<- yPredictGreater != voteLabelsT
      tempError <- sum(index * D)/sum(D)
      if(tempError < errorGreater) {
        errorGreater <- tempError
        minErrorGreaterThreshold <- tempCol[row]
      }
    }
    
    errorLess <- Inf
    minErrorLessThreshold <- Inf
    yPredictLess <- mat.or.vec(nrow(voteDataT), 1)
    #less
    for(row in 1:nrow(voteDataT)) {
      index <- tempCol < tempCol[row]
      yPredictLess[index] <- 1
      yPredictLess[!index] <- -1
      
      index<- yPredictLess != voteLabelsT
      tempError <- sum(index * D)/sum(D)
      if(tempError < errorLess) {
        errorLess <- tempError
        minErrorLessThreshold<- tempCol[row]
      }
    }
    
    stump<-matrix(1,5)
    if(errorLess >= errorGreater) {
      stump[1] <- col    #dimension
      stump[2] <- minErrorGreaterThreshold
      stump[3] <- errorGreater
      stump[4] <- -1  #less
      stump[5] <- 1 #greater
    } else {
      stump[1] <- col    #dimension
      stump[2] <- minErrorLessThreshold
      stump[3] <- errorLess
      stump[4] <- 1  #less
      stump[5] <- -1 #greater
    }
    if(minStumpError > stump[3]) {
      minErrorStump <- stump
      minStumpError<- stump[3]
    }
  }
  return(minErrorStump)
}

predStump <- function(voteDataP, stump) {
  column <- voteDataP[,stump[1]]
  index <- column >= stump[2]
  label <- mat.or.vec(nrow(voteDataP), 1)
  label[index] <- stump[5]
  label[!index] <- stump[4]
  return(label)
}


STARTING_PERCENT <- 5
PERCENT_TRAINING_SET_INCREMENT <- 2
startingNumberOfRows <- round(nrow(voteData) * STARTING_PERCENT / 100) 

numFolds <- 10
lockBinding("numFolds", globalenv())

index <- 1:nrow(voteData)
index <- sample(index) ### shuffle index
fold <- rep(1:numFolds, each=nrow(voteData)/numFolds)[1:nrow(voteData)]

folds <- split(index, fold) ### create list with indices for each fold

for(fold in 1:numFolds) {
 
  
  trainingSet <- voteData[-1 * unlist(folds[fold]),]
  validationSet <- voteData[unlist(folds[fold]),]
  
  rNames <- row.names(trainingSet) 
  sampRows <- sample(rNames, startingNumberOfRows)
  
  voteDataTraining <- subset(trainingSet, rNames %in% sampRows) 
  voteDataNotInTraining <- subset(trainingSet, !(rNames %in% sampRows)) 
  
  thresholdNumberOfRows <- round(nrow(voteData) / 2)
  
  while(TRUE) {
    trainingSetLabels <- as.matrix(data.frame(voteDataTraining[, c(ncol(voteDataTraining))]))
    trainingSetData <- as.matrix(voteDataTraining[, c(1:ncol(voteDataTraining) - 1)])
    
    class(trainingSetData) <- "numeric"
    class(trainingSetLabels) <- "numeric"
    
    errorsAb <- mat.or.vec(NUMITERATIONS, 1)
    D <- rep(1/nrow(trainingSetData), nrow(trainingSetData))
    weakClassifiers <- mat.or.vec(NUMITERATIONS, 5)
    weakClassifiersAlpha <- mat.or.vec(NUMITERATIONS, 1)
    for(i in 1:NUMITERATIONS) {
      weakClassifiers[i,] <- buildStump(trainingSetData, trainingSetLabels, D)
      weakClassifiersAlpha[i] <- 0.5 * log((1 - weakClassifiers[i, 3]) / max(1e-6, weakClassifiers[i, 3]))
      # update D
      weakClassifierPred <- predStump(trainingSetData, weakClassifiers[i,])
      tempD <- -1 * weakClassifiersAlpha[i] * (trainingSetLabels * weakClassifierPred)
      tempD <- D * exp(tempD)
      D <- tempD /sum(tempD)
      
      #make adaboost prediction
      WCLabels <- data.matrix(mat.or.vec(nrow(trainingSetData), i))
      for(wc in 1:i) {
        WCLabels[,wc] <- weakClassifiersAlpha[wc] * predStump(trainingSetData, weakClassifiers[wc,])
      }
      
      WCLabelsSums <- rowSums(WCLabels)
      
      index <- WCLabelsSums > 0
      abLabel <- mat.or.vec(nrow(trainingSetData), 1)
      abLabel[index] <- 1
      abLabel[!index] <- -1
      
      abErrorIdx <- abLabel != trainingSetLabels
      abError <- sum(abErrorIdx) / nrow(trainingSetData) 
    }
    
    # compute scores for data not used in training
    WCLabels <- data.matrix(mat.or.vec(nrow(voteDataNotInTraining), NUMITERATIONS))
    for(wc in 1:NUMITERATIONS) {
      WCLabels[,wc] <- weakClassifiersAlpha[wc] * predStump(voteDataNotInTraining, weakClassifiers[wc,])
    }
    
    WCLabelsSums <- rowSums(WCLabels) / sum(weakClassifiersAlpha)
    #add training data
    
    voteDataNotInTrainingTemp <- cbind(voteDataNotInTraining, WCLabelsSums)
    voteDataNotInTrainingTemp <- voteDataNotInTrainingTemp[ order(voteDataNotInTrainingTemp[,ncol(voteDataNotInTrainingTemp)]), ]
    
    
    numberOfNewRows <- round(nrow(voteDataNotInTraining) * PERCENT_TRAINING_SET_INCREMENT / 100)
    newRows <- voteDataNotInTrainingTemp[1:numberOfNewRows,1:ncol(voteDataNotInTrainingTemp) - 1]
    
    voteDataTraining <- rbind(voteDataTraining, newRows) 
    voteDataNotInTraining <- voteDataNotInTrainingTemp[numberOfNewRows:nrow(voteDataNotInTraining),1:ncol(voteDataNotInTrainingTemp) - 1]
    
    if(nrow(voteDataTraining) >= thresholdNumberOfRows) {
      break
    }
  }
  
  
  validationSetLabels <- as.matrix(data.frame(validationSet[, c(ncol(validationSet))]))
  validationSetData <- as.matrix(validationSet[, c(1:ncol(validationSet) - 1)])
  
  
  class(validationSetData) <- "numeric"
  class(validationSetLabels) <- "numeric"
  
  WCLabels <- data.matrix(mat.or.vec(nrow(validationSetData), i))
  for(wc in 1:i) {
    WCLabels[,wc] <- weakClassifiersAlpha[wc] * predStump(validationSetData, weakClassifiers[wc,])
  }
  
  WCLabelsSums <- rowSums(WCLabels)
  
  index <- WCLabelsSums > 0
  abLabel <- mat.or.vec(nrow(validationSetData), 1)
  abLabel[index] <- 1
  abLabel[!index] <- -1
  
  abErrorIdx <- abLabel != validationSetLabels
  abErrorValidation <- sum(abErrorIdx) / nrow(validationSetData) 
  
  
  
  print("fold")
  print(fold)
  #Training error
  print("training error")
  print(abError)
  #Testing error
  print("Testing error")
  print(abErrorValidation)
}
# 
# [1] "fold"
# [1] 1
# [1] "training error"
# [1] 0.03271028
# [1] "Testing error"
# [1] 0.02325581
# [1] "fold"
# [1] 2
# [1] "training error"
# [1] 0.03271028
# [1] "Testing error"
# [1] 0
# [1] "fold"
# [1] 3
# [1] "training error"
# [1] 0.02336449
# [1] "Testing error"
# [1] 0.06976744
# [1] "fold"
# [1] 4
# [1] "training error"
# [1] 0.009345794
# [1] "Testing error"
# [1] 0.02325581
# [1] "fold"
# [1] 5
# [1] "training error"
# [1] 0.02803738
# [1] "Testing error"
# [1] 0.02325581
# [1] "fold"
# [1] 6
# [1] "training error"
# [1] 0.01869159
# [1] "Testing error"
# [1] 0.06976744
# [1] "fold"
# [1] 7
# [1] "training error"
# [1] 0.03271028
# [1] "Testing error"
# [1] 0.04651163
# [1] "fold"
# [1] 8
# [1] "training error"
# [1] 0.03738318
# [1] "Testing error"
# [1] 0.02325581
# [1] "fold"
# [1] 9
# [1] "training error"
# [1] 0.01869159
# [1] "Testing error"
# [1] 0.04651163
# [1] "fold"
# [1] 10
# [1] "training error"
# [1] 0.04205607
# [1] "Testing error"
# [1] 0.02325581
