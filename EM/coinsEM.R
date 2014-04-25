rm(list=ls())

# number of coins
K <- 2

# coin tosses per selection
coinTossesPerSelection <- 10

# number of repetitions of picking/tossing the coin
M <- 3000

STOPTHRESHOLD <- 1e-10
MAX_ITER <- 10000

#probabilities for head for each coin
Q0 <- runif(K, 0.25, 0.75)  

#probability for picking a given coin
Pi <- rep(1/K, K)

coinTosses <- mat.or.vec(M, coinTossesPerSelection)

#Generate data
PiTrue<-runif(K)
PiTrue <- sort(PiTrue/sum(PiTrue))
Q0True <- runif(K, 0.25, 0.75) 
CumulativePiTrue <- mat.or.vec(1,K)
ChosenCoin <- mat.or.vec(M,1)

tempSum <- 0
for(i in 1:K) {
  tempSum <- tempSum + PiTrue[i]
  CumulativePiTrue[i] <- tempSum
}

for(i in 1:M) {
  pickCoin <- runif(1)
  coinIndex <- -1
  for(c in 1:K) {
    if(pickCoin <= CumulativePiTrue[c]) {
      coinIndex <- c
      break
    }
  }
  ChosenCoin[i] <- coinIndex
  coinTosses[i,] <- rbinom(coinTossesPerSelection, 1, Q0True[coinIndex])
}

## EM
newValue <- sum(Q0)

for(iteration in 1:MAX_ITER) {
  print("iteration")
  print(iteration)
  oldValue <- newValue
  Gamma <- mat.or.vec(M,K) 
  ExpectedHeads <- mat.or.vec(M,K)
  ExpectedTails <- mat.or.vec(M,K)
  for(i in 1:M) {
    Likelihoods <- mat.or.vec(1, K)
    for(j in 1:K) {
      numHeads <- sum(coinTosses[i,])
      Likelihoods[j] <- Pi[j] * Q0[j]^numHeads * (1 - Q0[j])^(coinTossesPerSelection - numHeads)
    }
    for(j in 1:K) {
      Gamma[i,j] <- Likelihoods[j] / sum(Likelihoods)
      ExpectedHeads[i,j] <- sum(coinTosses[i,]) * Gamma[i,j]
      ExpectedTails[i,j] <- (coinTossesPerSelection - sum(coinTosses[i,])) * Gamma[i,j]
    }
  }
  for(j in 1:K) {
    Q0[j] <- sum(ExpectedHeads[,j]) / (sum(ExpectedHeads[,j]) + sum(ExpectedTails[,j]))
  }
  
   for(j in 1:K) {
     Pi[j] <- sum(Gamma[,j]) / M
   }
#   
#   for(j in 1:K) {
#     tempSum <- 0
#     for(i in 1:M) {
#       tempSum <- tempSum + Gamma[i,j] * coinTosses[i,j]
#     }
#     Q0[j] <- tempSum / sum(Gamma[,j])
#   }
#   
  newValue <- sum(Q0)
  print("Error")
  error <- abs(newValue - oldValue)
  print(error)
  if(error < STOPTHRESHOLD) {
    break
  }
  
}



print("true and estimated data")
print(sort(Q0True))
print(sort(Q0))
print("true and estimated data")
print(sort(PiTrue))
print(sort(Pi))


#Sample output
# [1] "true and estimated data"
# > print(sort(Q0True))
# [1] 0.3460731 0.5531878
# > print(sort(Q0))
# [1] 0.3376885 0.5552361
# > print("true and estimated data")
# [1] "true and estimated data"
# > print(sort(PiTrue))
# [1] 0.4555262 0.5444738
# > print(sort(Pi))
# [1] 0.4716736 0.5283264


