rm(list=ls())

sigmoid2 = function(x) {
  if(x < 0.2) {
    return(0)
  } else if(x > 0.8) {
    return(1)
  }
  return(x)
}

sigmoid = function(x) {
  return(1.0 / (1.0 + exp(-x)))
}

learningRate = 0.2

W1.9 <- runif(8, -0.5, 0.5)
W1.10 <- runif(8, -0.5, 0.5)
W1.11 <- runif(8, -0.5, 0.5)

W2.9 <- runif(8, -0.5, 0.5)
W2.10 <- runif(8, -0.5, 0.5)
W2.11 <- runif(8, -0.5, 0.5)

W3.9 <- runif(8, -0.5, 0.5)
W3.10 <- runif(8, -0.5, 0.5)
W3.11 <- runif(8, -0.5, 0.5)

W4.9 <- runif(8, -0.5, 0.5)
W4.10 <- runif(8, -0.5, 0.5)
W4.11 <- runif(8, -0.5, 0.5)

W5.9 <- runif(8, -0.5, 0.5)
W5.10 <- runif(8, -0.5, 0.5)
W5.11 <- runif(8, -0.5, 0.5)

W6.9 <- runif(8, -0.5, 0.5)
W6.10 <- runif(8, -0.5, 0.5)
W6.11 <- runif(8, -0.5, 0.5)

W7.9 <- runif(8, -0.5, 0.5)
W7.10 <- runif(8, -0.5, 0.5)
W7.11 <- runif(8, -0.5, 0.5)

W8.9 <- runif(8, -0.5, 0.5)
W8.10 <- runif(8, -0.5, 0.5)
W8.11 <- runif(8, -0.5, 0.5)

W9.12 <- runif(1, -0.5, 0.5)
W9.13 <- runif(1, -0.5, 0.5)
W9.14 <- runif(1, -0.5, 0.5)
W9.15 <- runif(1, -0.5, 0.5)
W9.16 <- runif(1, -0.5, 0.5)
W9.17 <- runif(1, -0.5, 0.5)
W9.18 <- runif(1, -0.5, 0.5)
W9.19 <- runif(1, -0.5, 0.5)

W10.12 <- runif(1, -0.5, 0.5)
W10.13 <- runif(1, -0.5, 0.5)
W10.14 <- runif(1, -0.5, 0.5)
W10.15 <- runif(1, -0.5, 0.5)
W10.16 <- runif(1, -0.5, 0.5)
W10.17 <- runif(1, -0.5, 0.5)
W10.18 <- runif(1, -0.5, 0.5)
W10.19 <- runif(1, -0.5, 0.5)

W11.12 <- runif(1, -0.5, 0.5)
W11.13 <- runif(1, -0.5, 0.5)
W11.14 <- runif(1, -0.5, 0.5)
W11.15 <- runif(1, -0.5, 0.5)
W11.16 <- runif(1, -0.5, 0.5)
W11.17 <- runif(1, -0.5, 0.5)
W11.18 <- runif(1, -0.5, 0.5)
W11.19 <- runif(1, -0.5, 0.5)

theta9 <- runif(8, -0.5, 0.5)
theta10 <- runif(8, -0.5, 0.5)
theta11 <- runif(8, -0.5, 0.5)
theta12 <- runif(8, -0.5, 0.5)
theta13 <- runif(8, -0.5, 0.5)
theta14 <- runif(8, -0.5, 0.5)
theta15 <- runif(8, -0.5, 0.5)
theta16 <- runif(8, -0.5, 0.5)
theta17 <- runif(8, -0.5, 0.5)
theta18 <- runif(8, -0.5, 0.5)
theta19 <- runif(8, -0.5, 0.5)


X1 <- c(1,0,0,0,0,0,0,0)
X2 <- c(0,1,0,0,0,0,0,0)
X3 <- c(0,0,1,0,0,0,0,0)
X4 <- c(0,0,0,1,0,0,0,0)
X5 <- c(0,0,0,0,1,0,0,0)
X6 <- c(0,0,0,0,0,1,0,0)
X7 <- c(0,0,0,0,0,0,1,0)
X8 <- c(0,0,0,0,0,0,0,1)

counter <- 0
error <- list()
input <- t(cbind(X1, X2, X3, X4, X5, X6, X7, X8)) 
while(counter < 10000) {
  I9 <- X1 * W1.9 + X2 * W2.9 + X3 * W3.9 + X4 * W4.9 + X5 * W5.9 
  + X6 * W6.9 + X7 * W7.9 + X8 * W8.9 + theta9
  
  I10 <- X1 * W1.10 + X2 * W2.10 + X3 * W3.10 + X4 * W4.10 + X5 * W5.10 
  + X6 * W6.10 + X7 * W7.10 + X8 * W8.10 + theta10
  
  I11 <- X1 * W1.11 + X2 * W2.11 + X3 * W3.11 + X4 * W4.11 + X5 * W5.11 
  + X6 * W6.11 + X7 * W7.11 + X8 * W8.11 + theta11
  
  O9 <- I9
  O9[] <- vapply(I9, sigmoid, numeric(1))
  
  O10 <- I10
  O10[] <- vapply(I10, sigmoid, numeric(1))
  
  O11 <- I11
  O11[] <- vapply(I11, sigmoid, numeric(1))
  
  I12 <- W9.12 * O9 + W10.12 * O10 + W11.12 * O11 + theta12
  O12 <- I12
  O12[] <- vapply(I12, sigmoid, numeric(1))
  
  I13 <- W9.13 * O9 + W10.13 * O10 + W11.13 * O11 + theta13
  O13 <- I13
  O13[] <- vapply(I13, sigmoid, numeric(1))
  
  I14 <- W9.14 * O9 + W10.14 * O10 + W11.14 * O11 + theta14
  O14 <- I14
  O14[] <- vapply(I14, sigmoid, numeric(1))
  
  I15 <- W9.15 * O9 + W10.15 * O10 + W11.15 * O11 + theta15
  O15 <- I15
  O15[] <- vapply(I15, sigmoid, numeric(1))
  
  I16 <- W9.16 * O9 + W10.16 * O10 + W11.16 * O11 + theta16
  O16 <- I16
  O16[] <- vapply(I16, sigmoid, numeric(1))
  
  I17 <- W9.17 * O9 + W10.17 * O10 + W11.17 * O11 + theta17
  O17 <- I17
  O17[] <- vapply(I17, sigmoid, numeric(1))
  
  I18 <- W9.18 * O9 + W10.18 * O10 + W11.18 * O11 + theta18
  O18 <- I18
  O18[] <- vapply(I18, sigmoid, numeric(1))
  
  I19 <- W9.19 * O9 + W10.19 * O10 + W11.19 * O11 + theta19
  O19 <- I19
  O19[] <- vapply(I19, sigmoid, numeric(1))
  
  E12 <- O12 * (1 - O12) * (X1 - O12)
  E13 <- O13 * (1 - O13) * (X2 - O13)
  E14 <- O14 * (1 - O14) * (X3 - O14)
  E15 <- O15 * (1 - O15) * (X4 - O15)
  E16 <- O16 * (1 - O16) * (X5 - O16)
  E17 <- O17 * (1 - O17) * (X6 - O17)
  E18 <- O18 * (1 - O18) * (X7 - O18)
  E19 <- O19 * (1 - O19) * (X8 - O19)
  
  E9 <- O9 * (1 - O9) * (E12 * W9.12 + E13 * W9.13 + E14 * W9.14 + 
                           E15 * W9.15 + E16 * W9.16 + E16 * W9.17 + 
                           E17 * W9.17 + E18 * W9.18)
  
  E10 <- O10 * (1 - O10) * (E12 * W10.12 + E13 * W10.13 + E14 * W10.14 + 
                           E15 * W10.15 + E16 * W10.16 + E16 * W10.17 + 
                           E17 * W10.17 + E18 * W10.18)
  
  E11 <- O11 * (1 - O11) * (E12 * W11.12 + E13 * W11.13 + E14 * W11.14 + 
                           E15 * W11.15 + E16 * W11.16 + E16 * W11.17 + 
                           E17 * W11.17 + E18 * W11.18)
  
  # UPDATE
  W9.12 <- W9.12 + learningRate * E12 * O9
  W9.13 <- W9.13 + learningRate * E13 * O9
  W9.14 <- W9.14 + learningRate * E14 * O9
  W9.15 <- W9.15 + learningRate * E15 * O9
  W9.16 <- W9.16 + learningRate * E16 * O9
  W9.17 <- W9.17 + learningRate * E17 * O9
  W9.18 <- W9.18 + learningRate * E18 * O9
  W9.19 <- W9.19 + learningRate * E19 * O9
  
  W10.12 <- W10.12 + learningRate * E12 * O10
  W10.13 <- W10.13 + learningRate * E13 * O10
  W10.14 <- W10.14 + learningRate * E14 * O10
  W10.15 <- W10.15 + learningRate * E15 * O10
  W10.16 <- W10.16 + learningRate * E16 * O10
  W10.17 <- W10.17 + learningRate * E17 * O10
  W10.18 <- W10.18 + learningRate * E18 * O10
  W10.19 <- W10.19 + learningRate * E19 * O10
  
  W11.12 <- W11.12 + learningRate * E12 * O11
  W11.13 <- W11.13 + learningRate * E13 * O11
  W11.14 <- W11.14 + learningRate * E14 * O11
  W11.15 <- W11.15 + learningRate * E15 * O11
  W11.16 <- W11.16 + learningRate * E16 * O11
  W11.17 <- W11.17 + learningRate * E17 * O11
  W11.18 <- W11.18 + learningRate * E18 * O11
  W11.19 <- W11.19 + learningRate * E19 * O11
  
  W1.9 <- W1.9 + learningRate * E9 * X1
  W1.10 <- W1.10 + learningRate * E10 * X1
  W1.11 <- W1.11 + learningRate * E11 * X1
  
  W2.9 <- W2.9 + learningRate * E9 * X2
  W2.10 <- W2.10 + learningRate * E10 * X2
  W2.11 <- W2.11 + learningRate * E11 * X2
  
  W3.9 <- W3.9 + learningRate * E9 * X3
  W3.10 <- W3.10 + learningRate * E10 * X3
  W3.11 <- W3.11 + learningRate * E11 * X3
  
  W4.9 <- W4.9 + learningRate * E9 * X4
  W4.10 <- W4.10 + learningRate * E10 * X4
  W4.11 <- W4.11 + learningRate * E11 * X4
  
  W5.9 <- W5.9 + learningRate * E9 * X5
  W5.10 <- W5.10 + learningRate * E10 * X5
  W5.11 <- W5.11 + learningRate * E11 * X5
  
  W6.9 <- W6.9 + learningRate * E9 * X6
  W6.10 <- W6.10 + learningRate * E10 * X6
  W6.11 <- W6.11 + learningRate * E11 * X6
  
  W7.9 <- W7.9 + learningRate * E9 * X7
  W7.10 <- W7.10 + learningRate * E10 * X7
  W7.11 <- W7.11 + learningRate * E11 * X7
  
  W8.9 <- W8.9 + learningRate * E9 * X8
  W8.10 <- W8.10 + learningRate * E10 * X8
  W8.11 <- W8.11 + learningRate * E11 * X8
  
  theta12 <- theta12 + learningRate * E12 
  theta13 <- theta13 + learningRate * E13 
  theta14 <- theta14 + learningRate * E14 
  theta15 <- theta15 + learningRate * E15 
  theta16 <- theta16 + learningRate * E16 
  theta17 <- theta17 + learningRate * E17 
  theta18 <- theta18 + learningRate * E18 
  theta19 <- theta19 + learningRate * E19
  
  theta9 <- theta9 + learningRate * E9
  theta10 <- theta10 + learningRate * E10
  theta11 <- theta11 + learningRate * E11
  
  result <- t(cbind(O12, O13, O14, O15, O16, O17, O18, O19)) 
  error[counter] <- sum(input - result)

  
  
  counter <- counter + 1
}  
  
  result <- t(cbind(O12, O13, O14, O15, O16, O17, O18, O19)) 
  result2 <- result
  result2[] <- vapply(result, sigmoid2, numeric(1))