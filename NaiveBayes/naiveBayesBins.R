rm(list=ls())
library(matrixStats)

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
  
  trainingSetMatrix <- rbind(trainingSetNegative_dataMatrix,trainingSetPositive_dataMatrix)
  
  
  minValue <- colMins(trainingSetMatrix)
  overallMeanValue <- colMeans(trainingSetMatrix)
  maxValue <- colMaxs(trainingSetMatrix)
  
  spamMeanValue <- colMeans(trainingSetPositive_dataMatrix)
  nonSpamMeanValue <- colMeans(trainingSetNegative_dataMatrix)
  
  highMeanValue <- pmax(spamMeanValue, nonSpamMeanValue)
  lowMeanValue <- pmin(spamMeanValue, nonSpamMeanValue)
  
  probabilityPositive <- nrow(trainingSetPositive_dataMatrix) / nrow(trainingSet)
  probabilityNegative <- nrow(trainingSetNegative_dataMatrix) / nrow(trainingSet)
  
  positiveProbabilitiesPerFeature <-  mat.or.vec(ncol(trainingSetPositive_dataMatrix), 4)
  negativeProbabilitiesPerFeature <-  mat.or.vec(ncol(trainingSetNegative_dataMatrix), 4)
  for(feature in 1:ncol(trainingSetPositive_dataMatrix)) {
    firstBinPositive <- subset(trainingSetPositive_dataMatrix, trainingSetPositive_dataMatrix[,feature] >= minValue[feature] & trainingSetPositive_dataMatrix[,feature] <= lowMeanValue[feature])
    secondBinPositive <- subset(trainingSetPositive_dataMatrix, trainingSetPositive_dataMatrix[,feature] > lowMeanValue[feature] & trainingSetPositive_dataMatrix[,feature] <= overallMeanValue[feature])
    thirdBinPositive <- subset(trainingSetPositive_dataMatrix, trainingSetPositive_dataMatrix[,feature] > overallMeanValue[feature] & trainingSetPositive_dataMatrix[,feature] <= highMeanValue[feature])
    fourthBinPositive <- subset(trainingSetPositive_dataMatrix, trainingSetPositive_dataMatrix[,feature] > highMeanValue[feature] & trainingSetPositive_dataMatrix[,feature] <= maxValue[feature])
    
    firstBinNegative <- subset(trainingSetNegative_dataMatrix, trainingSetNegative_dataMatrix[,feature] >= minValue[feature] & trainingSetNegative_dataMatrix[,feature] <= lowMeanValue[feature])
    secondBinNegative <- subset(trainingSetNegative_dataMatrix, trainingSetNegative_dataMatrix[,feature] > lowMeanValue[feature] & trainingSetNegative_dataMatrix[,feature] <= overallMeanValue[feature])
    thirdBinNegative <- subset(trainingSetNegative_dataMatrix, trainingSetNegative_dataMatrix[,feature] > overallMeanValue[feature] & trainingSetNegative_dataMatrix[,feature] <= highMeanValue[feature])
    fourthBinNegative <- subset(trainingSetNegative_dataMatrix, trainingSetNegative_dataMatrix[,feature] > highMeanValue[feature] & trainingSetNegative_dataMatrix[,feature] <= maxValue[feature])

    positiveProbabilitiesPerFeature[feature, 1] <- positiveProbabilitiesPerFeature[feature, 1] + nrow(firstBinPositive) + 1
    positiveProbabilitiesPerFeature[feature, 1] <- positiveProbabilitiesPerFeature[feature, 1] / (2 + nrow(trainingSetPositive_dataMatrix)) 
    negativeProbabilitiesPerFeature[feature, 1] <- negativeProbabilitiesPerFeature[feature, 1] + nrow(firstBinNegative) + 1
    negativeProbabilitiesPerFeature[feature, 1] <- negativeProbabilitiesPerFeature[feature, 1] / (2 + nrow(trainingSetNegative_dataMatrix)) 
    
    positiveProbabilitiesPerFeature[feature, 2] <- positiveProbabilitiesPerFeature[feature, 2] + nrow(secondBinPositive) + 1
    positiveProbabilitiesPerFeature[feature, 2] <- positiveProbabilitiesPerFeature[feature, 2] / (2 + nrow(trainingSetPositive_dataMatrix)) 
    negativeProbabilitiesPerFeature[feature, 2] <- negativeProbabilitiesPerFeature[feature, 2] + nrow(secondBinNegative) + 1
    negativeProbabilitiesPerFeature[feature, 2] <- negativeProbabilitiesPerFeature[feature, 2] / (2 + nrow(trainingSetNegative_dataMatrix)) 
    
    positiveProbabilitiesPerFeature[feature, 3] <- positiveProbabilitiesPerFeature[feature, 3] + nrow(thirdBinPositive) + 1
    positiveProbabilitiesPerFeature[feature, 3] <- positiveProbabilitiesPerFeature[feature, 3] / (2 + nrow(trainingSetPositive_dataMatrix)) 
    negativeProbabilitiesPerFeature[feature, 3] <- negativeProbabilitiesPerFeature[feature, 3] + nrow(thirdBinNegative) + 1
    negativeProbabilitiesPerFeature[feature, 3] <- negativeProbabilitiesPerFeature[feature, 3] / (2 + nrow(trainingSetNegative_dataMatrix)) 
    
    positiveProbabilitiesPerFeature[feature, 4] <- positiveProbabilitiesPerFeature[feature, 4] + nrow(fourthBinPositive) + 1
    positiveProbabilitiesPerFeature[feature, 4] <- positiveProbabilitiesPerFeature[feature, 4] / (2 + nrow(trainingSetPositive_dataMatrix)) 
    negativeProbabilitiesPerFeature[feature, 4] <- negativeProbabilitiesPerFeature[feature, 4] + nrow(fourthBinNegative) + 1
    negativeProbabilitiesPerFeature[feature, 4] <- negativeProbabilitiesPerFeature[feature, 4] / (2 + nrow(trainingSetNegative_dataMatrix)) 
    
  }

  #TEST on TRAINING
  trainingSetPositiveProbabilities <- mat.or.vec(nrow(trainingSetPositive_dataMatrix), ncol(trainingSetPositive_dataMatrix))
  predictedPositiveProbabilityFinal <- mat.or.vec(nrow(trainingSetPositiveProbabilities), 1)
  for(row in 1:nrow(trainingSetPositive_dataMatrix)) {
    for(col in 1:ncol(trainingSetPositive_dataMatrix)) {
      if(trainingSetPositive_dataMatrix[row,col] >= minValue[col] & trainingSetPositive_dataMatrix[row, col] <= lowMeanValue[col]) {
        trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 1]
      } else {
        if(trainingSetPositive_dataMatrix[row, col] > lowMeanValue[col] & trainingSetPositive_dataMatrix[row, col] <= overallMeanValue[col]) {
          trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 2]
        } else {
          if(trainingSetPositive_dataMatrix[row, col] > overallMeanValue[col] & trainingSetPositive_dataMatrix[row, col] <= highMeanValue[col]) {
            trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 3]
          }
          else {
            if(trainingSetPositive_dataMatrix[row, col] > highMeanValue[col] & trainingSetPositive_dataMatrix[row, col] <= maxValue[col]) {
              trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 4]
            }
          }
        }
      }
    }
    predictedPositiveProbabilityFinal[row] <- sum(log(trainingSetPositiveProbabilities[row,])) + log(probabilityPositive)
  }
  
  trainingSetNegativeProbabilities <- mat.or.vec(nrow(trainingSetPositive_dataMatrix), ncol(trainingSetPositive_dataMatrix))
  predictedNegativeProbabilityFinal <- mat.or.vec(nrow(trainingSetNegativeProbabilities), 1)
  for(row in 1:nrow(trainingSetPositive_dataMatrix)) {
    for(col in 1:ncol(trainingSetPositive_dataMatrix)) {
      if(trainingSetPositive_dataMatrix[row,col] >= minValue[col] & trainingSetPositive_dataMatrix[row, col] <= lowMeanValue[col]) {
        trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 1]
      } else {
        if(trainingSetPositive_dataMatrix[row, col] > lowMeanValue[col] & trainingSetPositive_dataMatrix[row, col] <= overallMeanValue[col]) {
          trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 2]
        } else {
          if(trainingSetPositive_dataMatrix[row, col] > overallMeanValue[col] & trainingSetPositive_dataMatrix[row, col] <= highMeanValue[col]) {
            trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 3]
            } else {
              if(trainingSetPositive_dataMatrix[row, col] > highMeanValue[col] & trainingSetPositive_dataMatrix[row, col] <= maxValue[col]) {
                trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 4]
              }
            }
        }
      }
    }
    predictedNegativeProbabilityFinal[row] <- sum(log(trainingSetNegativeProbabilities[row,])) + log(probabilityNegative)
  }
  
  TP <- sum(predictedPositiveProbabilityFinal > predictedNegativeProbabilityFinal)
  FN <-  length(predictedPositiveProbabilityFinal) - TP
  
  trainingSetPositiveProbabilities <- mat.or.vec(nrow(trainingSetNegative_dataMatrix), ncol(trainingSetNegative_dataMatrix))
  predictedPositiveProbabilityFinal <- mat.or.vec(nrow(trainingSetPositiveProbabilities), 1)
  for(row in 1:nrow(trainingSetNegative_dataMatrix)) {
    for(col in 1:ncol(trainingSetNegative_dataMatrix)) {
      if(trainingSetNegative_dataMatrix[row,col] >= minValue[col] & trainingSetNegative_dataMatrix[row, col] <= lowMeanValue[col]) {
        trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 1]
      } else {
        if(trainingSetNegative_dataMatrix[row, col] > lowMeanValue[col] & trainingSetNegative_dataMatrix[row, col] <= overallMeanValue[col]) {
          trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 2]
        } else {
          if(trainingSetNegative_dataMatrix[row, col] > overallMeanValue[col] & trainingSetNegative_dataMatrix[row, col] <= highMeanValue[col]) {
            trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 3]
          }
          else {
            if(trainingSetNegative_dataMatrix[row, col] > highMeanValue[col] & trainingSetNegative_dataMatrix[row, col] <= maxValue[col]) {
              trainingSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 4]
            }
          }
        }
      }
    }
    predictedPositiveProbabilityFinal[row] <- sum(log(trainingSetPositiveProbabilities[row,])) + log(probabilityPositive)
  }
  
  trainingSetNegativeProbabilities <- mat.or.vec(nrow(trainingSetNegative_dataMatrix), ncol(trainingSetNegative_dataMatrix))
  predictedNegativeProbabilityFinal <- mat.or.vec(nrow(trainingSetNegativeProbabilities), 1)
  for(row in 1:nrow(trainingSetNegative_dataMatrix)) {
    for(col in 1:ncol(trainingSetNegative_dataMatrix)) {
      if(trainingSetNegative_dataMatrix[row,col] >= minValue[col] & trainingSetNegative_dataMatrix[row, col] <= lowMeanValue[col]) {
        trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 1]
      } else {
        if(trainingSetNegative_dataMatrix[row, col] > lowMeanValue[col] & trainingSetNegative_dataMatrix[row, col] <= overallMeanValue[col]) {
          trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 2]
        } else {
          if(trainingSetNegative_dataMatrix[row, col] > overallMeanValue[col] & trainingSetNegative_dataMatrix[row, col] <= highMeanValue[col]) {
            trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 3]
          } else {
            if(trainingSetNegative_dataMatrix[row, col] > highMeanValue[col] & trainingSetNegative_dataMatrix[row, col] <= maxValue[col]) {
              trainingSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 4]
            }
          }
        }
      }
    }
    predictedNegativeProbabilityFinal[row] <- sum(log(trainingSetNegativeProbabilities[row,])) + log(probabilityNegative)
  }
  
  TN <- sum(predictedPositiveProbabilityFinal < predictedNegativeProbabilityFinal)
  FP <- length(predictedPositiveProbabilityFinal) - TN
  
  trainAccuracy <- trainAccuracy +  (TP + TN)/(TP+TN+FP+FN)
  
  #TEST on Validation set
  validationSetPositiveProbabilities <- mat.or.vec(nrow(validationSetPositive_dataMatrix), ncol(validationSetPositive_dataMatrix))
  predictedPositiveProbabilityFinal <- mat.or.vec(nrow(validationSetPositiveProbabilities), 1)
  for(row in 1:nrow(validationSetPositive_dataMatrix)) {
    for(col in 1:ncol(validationSetPositive_dataMatrix)) {
      if(validationSetPositive_dataMatrix[row,col] >= minValue[col] & validationSetPositive_dataMatrix[row, col] <= lowMeanValue[col]) {
        validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 1]
      } else {
        if(validationSetPositive_dataMatrix[row, col] > lowMeanValue[col] & validationSetPositive_dataMatrix[row, col] <= overallMeanValue[col]) {
          validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 2]
        } else {
          if(validationSetPositive_dataMatrix[row, col] > overallMeanValue[col] & validationSetPositive_dataMatrix[row, col] <= highMeanValue[col]) {
            validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 3]
          }
          else {
            if(validationSetPositive_dataMatrix[row, col] > highMeanValue[col] & validationSetPositive_dataMatrix[row, col] <= maxValue[col]) {
              validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 4]
            }
          }
        }
      }
    }
    predictedPositiveProbabilityFinal[row] <- sum(log(validationSetPositiveProbabilities[row,])) + log(probabilityPositive)
  }
  
  validationSetNegativeProbabilities <- mat.or.vec(nrow(validationSetPositive_dataMatrix), ncol(validationSetPositive_dataMatrix))
  predictedNegativeProbabilityFinal <- mat.or.vec(nrow(validationSetNegativeProbabilities), 1)
  for(row in 1:nrow(validationSetPositive_dataMatrix)) {
    for(col in 1:ncol(validationSetPositive_dataMatrix)) {
      if(validationSetPositive_dataMatrix[row,col] >= minValue[col] & validationSetPositive_dataMatrix[row, col] <= lowMeanValue[col]) {
        validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 1]
      } else {
        if(validationSetPositive_dataMatrix[row, col] > lowMeanValue[col] & validationSetPositive_dataMatrix[row, col] <= overallMeanValue[col]) {
          validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 2]
        } else {
          if(validationSetPositive_dataMatrix[row, col] > overallMeanValue[col] & validationSetPositive_dataMatrix[row, col] <= highMeanValue[col]) {
            validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 3]
          } else {
            if(validationSetPositive_dataMatrix[row, col] > highMeanValue[col] & validationSetPositive_dataMatrix[row, col] <= maxValue[col]) {
              validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 4]
            }
          }
        }
      }
    }
    predictedNegativeProbabilityFinal[row] <- sum(log(validationSetNegativeProbabilities[row,])) + log(probabilityNegative)
  }
  
  TP <- sum(predictedPositiveProbabilityFinal > predictedNegativeProbabilityFinal)
  FN <-  length(predictedPositiveProbabilityFinal) - TP
  
  validationSetPositiveProbabilities <- mat.or.vec(nrow(validationSetNegative_dataMatrix), ncol(validationSetNegative_dataMatrix))
  predictedPositiveProbabilityFinal <- mat.or.vec(nrow(validationSetPositiveProbabilities), 1)
  for(row in 1:nrow(validationSetNegative_dataMatrix)) {
    for(col in 1:ncol(validationSetNegative_dataMatrix)) {
      if(validationSetNegative_dataMatrix[row,col] >= minValue[col] & validationSetNegative_dataMatrix[row, col] <= lowMeanValue[col]) {
        validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 1]
      } else {
        if(validationSetNegative_dataMatrix[row, col] > lowMeanValue[col] & validationSetNegative_dataMatrix[row, col] <= overallMeanValue[col]) {
          validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 2]
        } else {
          if(validationSetNegative_dataMatrix[row, col] > overallMeanValue[col] & validationSetNegative_dataMatrix[row, col] <= highMeanValue[col]) {
            validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 3]
          }
          else {
            if(validationSetNegative_dataMatrix[row, col] > highMeanValue[col] & validationSetNegative_dataMatrix[row, col] <= maxValue[col]) {
              validationSetPositiveProbabilities[row, col] <- positiveProbabilitiesPerFeature[col, 4]
            }
          }
        }
      }
    }
    predictedPositiveProbabilityFinal[row] <- sum(log(validationSetPositiveProbabilities[row,])) + log(probabilityPositive)
  }
  
  validationSetNegativeProbabilities <- mat.or.vec(nrow(validationSetNegative_dataMatrix), ncol(validationSetNegative_dataMatrix))
  predictedNegativeProbabilityFinal <- mat.or.vec(nrow(validationSetNegativeProbabilities), 1)
  for(row in 1:nrow(validationSetNegative_dataMatrix)) {
    for(col in 1:ncol(validationSetNegative_dataMatrix)) {
      if(validationSetNegative_dataMatrix[row,col] >= minValue[col] & validationSetNegative_dataMatrix[row, col] <= lowMeanValue[col]) {
        validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 1]
      } else {
        if(validationSetNegative_dataMatrix[row, col] > lowMeanValue[col] & validationSetNegative_dataMatrix[row, col] <= overallMeanValue[col]) {
          validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 2]
        } else {
          if(validationSetNegative_dataMatrix[row, col] > overallMeanValue[col] & validationSetNegative_dataMatrix[row, col] <= highMeanValue[col]) {
            validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 3]
          } else {
            if(validationSetNegative_dataMatrix[row, col] > highMeanValue[col] & validationSetNegative_dataMatrix[row, col] <= maxValue[col]) {
              validationSetNegativeProbabilities[row, col] <- negativeProbabilitiesPerFeature[col, 4]
            }
          }
        }
      }
    }
    predictedNegativeProbabilityFinal[row] <- sum(log(validationSetNegativeProbabilities[row,])) + log(probabilityNegative)
  }
  
  TN <- sum(predictedPositiveProbabilityFinal < predictedNegativeProbabilityFinal)
  FP <- length(predictedPositiveProbabilityFinal) - TN
  
  testAccuracy <- trainAccuracy +  (TP + TN)/(TP+TN+FP+FN)
  

}

trainAccuracy <- trainAccuracy / numFolds
testAccuracy <- testAccuracy / numFolds

print("Average train error:")
print(1-trainAccuracy)
print("Average test error:")
print(1-testAccuracy)