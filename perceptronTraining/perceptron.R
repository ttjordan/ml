rm(list = ls())

sigmoid = function(x) {
  if(x <= 0) {
    return(-1)
  }
  return(1)
}

equalityFunction = function(x) {
  if(x == 0) {
    return(0)
  }
  return(1)
}

perceptronData = read.table("/Users/tjordan/Desktop/ML/Data/perceptronData.txt")

W <- rbind(c(1,runif(4, -0.05, 0.05)))

XTrainingData <- perceptronData[, c(1:ncol(perceptronData) - 1)]
YTrainingData <- perceptronData[, c(ncol(perceptronData))]

XTrainingMatrix <- data.matrix(cbind(rep(1,length(YTrainingData)), XTrainingData))
YTrainingMatrix <- data.matrix(YTrainingData)

totalMistakes <- 0
learningRate = 0.1
iteration <- 1

while(TRUE) {
  newMistake <- 0
  for(i in 1:length(YTrainingData)) {
    predictedValue = sigmoid(W %*% XTrainingMatrix[i,])
    if(YTrainingMatrix[i,] != predictedValue) {
      newMistake <- newMistake + 1
      W <- W + 2 * learningRate * YTrainingMatrix[i,] * XTrainingMatrix[i,] 
    }
  }
  
  if(newMistake == 0) {
    break
  } else {
    totalMistakes <- totalMistakes + newMistake
  }
  
  print(paste("Iteration: ", iteration))
  print(paste("Total mistakes: ", totalMistakes))
  iteration <- iteration + 1
}

print("Classifier weights: ")
print(W)

print("Normalized classifier")
print(W/W[1])