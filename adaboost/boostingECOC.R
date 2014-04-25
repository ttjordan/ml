rm(list=ls())
library('Matrix')
library('ada')

trainingSetFull <- read.csv("C:/Users/Tad/Desktop/20newsgroup/parsed_train.txt", header=FALSE)
validationSet <- read.csv("C:/Users/Tad/Desktop/20newsgroup/parsed_test.txt", header=FALSE)

PERCENT <- 40
n <- PERCENT * nrow(trainingSetFull) / 100
trainingSet <- trainingSetFull[sample(nrow(trainingSetFull), n),]

ecocCodes <-read.csv("C:/Users/Tad/Desktop/20newsgroup/ecoccodes.csv", header=FALSE)
 
NUMITERATIONS <- 200

buildStump <- function(voteData, voteLabels, D) {
  numCols <- ncol(voteData)
  
  minStumpError <- Inf
  minErrorStump<-matrix(1,5)
  for(col in 1:numCols) {
    tempCol <-  data.matrix(voteData[,col])
    
    errorGreater <- Inf
    minErrorGreaterThreshold <- Inf
    yPredictGreater <- mat.or.vec(nrow(voteData), 1)
    #greater
    for(row in 1:nrow(voteData)) {
      index <- tempCol >= tempCol[row]
      yPredictGreater[index] <- 1
      yPredictGreater[!index] <- -1
      
      index<- yPredictGreater != voteLabels
      tempError <- sum(index * D)/sum(D)
      if(tempError < errorGreater) {
        errorGreater <- tempError
        minErrorGreaterThreshold <- tempCol[row]
      }
    }
    
    errorLess <- Inf
    minErrorLessThreshold <- Inf
    yPredictLess <- mat.or.vec(nrow(voteData), 1)
    #less
    for(row in 1:nrow(voteData)) {
      index <- tempCol < tempCol[row]
      yPredictLess[index] <- 1
      yPredictLess[!index] <- -1
      
      index<- yPredictLess != voteLabels
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

predStump <- function(voteData, stump) {
  column <- voteData[,stump[1]]
  index <- column >= stump[2]
  label <- mat.or.vec(nrow(voteData), 1)
  label[index] <- stump[5]
  label[!index] <- stump[4]
  return(label)
}


numberOfClassifiers <- ncol(ecocCodes)
numberOfClasses <- 8

OriginalLabels <- trainingSet[,1]
TrainingLabels <- mat.or.vec(nrow(trainingSet), numberOfClassifiers)


abClassifiers <- list()
abClassifiersWeights <- list()


trainingSetLabels <- data.frame(trainingSet[, 1])
trainingSetData <- trainingSet[, c(1:1:ncol(trainingSet))]

class(trainingSetData) <- "numeric"
class(trainingSetLabels) <- "numeric"

#trainingSetData <- Matrix(data.matrix(trainingSetData), sparse = TRUE)


for(i in 1:numberOfClassifiers) {
  print("Training classifier")
  print(i)
  #get labels for classifier i
  newTrainingLabels <- rep(-1,nrow(trainingSet))
  
  for(j in 1:numberOfClasses) {
    if(ecocCodes[i,j] == 1) {
      newTrainingLabels[OriginalLabels == j - 1] <- 1
    } 
  }
  trainingSetWithLabels <- cbind(newTrainingLabels,trainingSetData)
  
  errorsAb <- mat.or.vec(NUMITERATIONS, 1)
  D <- rep(1/nrow(trainingSet), nrow(trainingSet))
  weakClassifiers <- mat.or.vec(NUMITERATIONS, 5)
  weakClassifiersAlpha <- mat.or.vec(NUMITERATIONS, 1)
  for(iter in 1:NUMITERATIONS) {
    print("Training classifier")
    print(i)
    print("iter")
    print(iter)
    
    trainingDataWithLabels <- cbind(trainingSetData,newTrainingLabels)
    gdis<-ada(,data=trainingSetData,iter=20,nu=1,type="discrete")
    
    weakClassifiers[iter,] <- buildStump(trainingSetData, newTrainingLabels, D)
    weakClassifiersAlpha[iter] <- 0.5 * log((1 - weakClassifiers[iter, 3]) / weakClassifiers[iter, 3])
    # update D
    weakClassifierPred <- predStump(trainingSetData, weakClassifiers[iter,])
    tempD <- -1 * weakClassifiersAlpha[iter] * (newTrainingLabels * weakClassifierPred)
    tempD <- D * exp(tempD)
    D <- tempD / sum(tempD)
  }
  
  abClassifiers[[i]] <- weakClassifiers
  abClassifiersWeights[[i]] <- weakClassifiersAlpha
}

validationSetLabels <- as.matrix(data.frame(validationSet[, 1]))
validationSetData <- as.matrix(validationSet[, c(1:1:ncol(validationSet))])


class(validationSetData) <- "numeric"
class(validationSetLabels) <- "numeric"


Labels <- mat.or.vec(nrow(validationSetData), numberOfClassifiers)
for i in 1:numberOfClassifiers {
  WCLabels <- data.matrix(mat.or.vec(nrow(validationSetData), i))
  weakClassifiersAlpha <- abClassifiersWeights[[i]]
  weakClassifiers <-  abClassifiers[[i]]
  for(wc in 1:NUMITERATIONS) {
    WCLabels[,wc] <- weakClassifiersAlpha[wc] * predStump(validationSetData, weakClassifiers[wc,])
  }
  
  WCLabelsSums <- rowSums(WCLabels)
  
  index <- WCLabelsSums > 0
  abLabel <- mat.or.vec(nrow(validationSetData), 1)
  abLabel[index] <- 1
  abLabel[!index] <- -1 
  Labels[,i] <- abLabel
}


FinalLabels <- mat.or.vec(nrow(validationSetData))
for(row in 1:nrow(validationSetData)) {
  closestClass <- Inf
  closestDiff <- Inf
  for(class in 1:numberOfClasses) {  
    tempDistance <- sum(Labels[row,] != ecocCodes[class,])
    if(tempDistance < closestDiff) {
      closestDiff <- tempSum
      closestClass <- class - 1
    }
  }
  FinalLabels[row] <- closestClass
}


abErrorIdx <- FinalLabels != validationSetLabels
abErrorValidation <- sum(abErrorIdx) / nrow(validationSetData) 


#Testing error
print("Testing error")
print(abErrorValidation)




