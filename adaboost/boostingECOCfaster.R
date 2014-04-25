rm(list=ls())
library('ada')

trainingSetFull <- read.csv("C:/Users/Tad/Desktop/20newsgroup/parsed_train.txt", header=FALSE)
validationSet <- read.csv("C:/Users/Tad/Desktop/20newsgroup/parsed_test.txt", header=FALSE)

PERCENT <- 15
n <- PERCENT * nrow(trainingSetFull) / 100
trainingSet <- trainingSetFull[sample(nrow(trainingSetFull), n),]

ecocCodes <-read.csv("C:/Users/Tad/Desktop/20newsgroup/ecoccodes.csv", header=FALSE)


numberOfClassifiers <- ncol(ecocCodes)
numberOfClasses <- 8

OriginalLabels <- trainingSet[,1]
TrainingLabels <- mat.or.vec(nrow(trainingSet), numberOfClassifiers)


abClassifiers <- list()


trainingSetLabels <- data.frame(trainingSet[, 1])
trainingSetData <- trainingSet[, c(1:1:ncol(trainingSet))]



#trainingSetData <- Matrix(data.matrix(trainingSetData), sparse = TRUE)


for(i in 1:numberOfClassifiers) {
  print("Training classifier")
  print(i)
  
  #get labels for classifier i
  newTrainingLabels <- rep(-1,nrow(trainingSet))
  for(j in 1:numberOfClasses) {
    if(ecocCodes[j,i] == 1) {
      newTrainingLabels[OriginalLabels == (j - 1)] <- 1
    } 
  }
  trainingSetWithLabels <- cbind(newTrainingLabels,trainingSetData)
  
  gdis<-ada(newTrainingLabels~.,data=trainingSetWithLabels,iter=50,nu=1,type="discrete")
  abClassifiers[[i]] <- gdis
}

validationSetLabels <- as.matrix(data.frame(validationSet[, 1]))
validationSetData <- as.matrix(validationSet[, c(1:1:ncol(validationSet))])

Labels <- mat.or.vec(nrow(validationSetData), numberOfClassifiers)

for(i in 1:numberOfClassifiers) {
  abLabels <- predict( abClassifiers[[i]],validationSetData, type=c("vector"))
  Labels[,i] <- abLabels
}


FinalLabels <- mat.or.vec(nrow(validationSetData),1)
for(row in 1:nrow(validationSetData)) {
  closestClass <- Inf
  closestDiff <- Inf
  for(class in 1:numberOfClasses) {  
    tempDistance <- sum(Labels[row,] != ecocCodes[class,])
    if(tempDistance < closestDiff) {
      closestDiff <- tempDistance
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




  

  
  
  
  