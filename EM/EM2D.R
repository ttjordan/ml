rm(list=ls())
library(matrixStats)
library(mvtnorm)

#number of dimensions for Gaussian
n <- 2

gaussiansData <- read.table("/Users/tjordan/Desktop/ML/Data/2gaussian.txt", header=FALSE)

numObservations <- nrow(gaussiansData)
lockBinding("numObservations", globalenv())

#number of Gaussians
K <- 2

Mu <- mat.or.vec(K, n)
W <- mat.or.vec(K,1)

clusters <- kmeans(gaussiansData, K, iter.max=50)
gaussiansDataWithLabels <- cbind(gaussiansData, clusters$cluster)

COVS <- list()
for(i in 1:K) {
  temp <- subset(gaussiansDataWithLabels, clusters$cluster==1)
  W[i] <- nrow(temp)/nrow(gaussiansDataWithLabels) 
  COVS[[i]] <- cov(temp[,1:2])
}

for(i in 1:K) {
  Mu[i,] <- clusters$centers[i,]
}

STOPTHRESHOLD <- 1e-5
MAX_ITER <- 1000

#For testing 3gaussians
#W<-c(0.3206959273694992, 0.45357304148330913, 0.22573103114719137)
#Mu <- rbind(c(6.95145515, 4.2786734), c(4.93326615, 7.06691818), c(3.27472168, 3.19007281))
#COVS <- list(matrix(c(1.05843789, 0.20623345,0.20623345, 1.33288348 ),2,2), matrix(c(0.97443921, 0.25982045,0.25982045, 0.94619833 ),2,2), matrix(c(1.44474827, 0.33745813,0.33745813, 3.69796638 ),2,2))



sum <- 0
for(i in 1:numObservations) {
  tempSum <- 0
  for(j in 1:K) {
    tempSum <- tempSum + W[j] * dmvnorm(gaussiansData[i,], Mu[j,], COVS[[j]], log=FALSE)
  }
  sum <- sum + log(tempSum)
}
newLikelihood <- sum/numObservations


for(iteration in 1:100) {
  print("iteration")
  print(iteration)
  oldLikelihood <- newLikelihood
  gamma <- mat.or.vec(numObservations, K)
  
  for(i in 1:numObservations) {
    sum <- 0
    for(x in 1:K) {
      sum <- sum + W[x] * dmvnorm(gaussiansData[i,], Mu[x,], COVS[[x]], log=FALSE)
    }
    for(j in 1:K) {
      gamma[i,j] <- W[j] * dmvnorm(gaussiansData[i,], Mu[j,], COVS[[j]], log=FALSE)/ sum
    }
  }
  
  N <- mat.or.vec(1, K)
  for(j in 1:K) {
    N[j] <- sum(gamma[,j])
  }
  
  #MAXIMIZATION
  for(j in 1:K) {
    W[j] <- N[j] / numObservations
  }
  
  for(j in 1:K) {
      Mu[j,] <- colSums(gamma[,j]*gaussiansData) / N[j]
  }
 
  for(j in 1:K) {
    tempCov <- mat.or.vec(n, n)
    for(i in 1:numObservations) {
      tempCov <- tempCov + t(data.matrix(gamma[i,j] * (gaussiansData[i,] - Mu[j,]))) %*% data.matrix(gaussiansData[i,] - Mu[j,])
    }
    tempCov <- tempCov /  N[j]
    COVS[[j]] <- tempCov
  }
  
  sum <- 0
  for(i in 1:numObservations) {
    tempSum <- 0
    for(j in 1:K) {
      tempSum <- tempSum + W[j] * dmvnorm(gaussiansData[i,], Mu[j,], COVS[[j]], log=FALSE)
    }
    sum <- sum + log(tempSum)
  }
  newLikelihood <- sum/numObservations
  
  diff <- newLikelihood - oldLikelihood
  if(abs(diff) < STOPTHRESHOLD * abs(newLikelihood)) {
    break
  }
  print("likelihood diff")
  print(diff)
}

labels <- mat.or.vec(numObservations, 1)
for(i in 1:numObservations) {
  label <- -1
  largestProb <- -999999
  for(j in 1:K) {
    candidate <- dmvnorm(gaussiansData[i,], Mu[j,], COVS[[j]], log=FALSE)
    if(candidate > largestProb) {
      largestProb <- candidate
      label <- j
    }
  }
  labels[i] <- label
}



