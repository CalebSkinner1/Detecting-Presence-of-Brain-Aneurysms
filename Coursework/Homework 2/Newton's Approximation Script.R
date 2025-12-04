# Script for Newton's Method Problem 1.5

X <- matrix(c(1, 1, 1, 1, 0, 1, 0, 1, 3, 3, 1, 1), nrow = 4, byrow = FALSE)
y <- c(1, 1, 0, 0)
theta <- c(0, -2, 1)
lambda <- 0.07
m <- length(y)

iter <- 2
for(i in 1:iter){
  y_hat <- as.vector(1/(1 + exp(-X%*%theta))) # probabilities inferred from theta
  S <- diag(y_hat*(1-y_hat)) # diagonal S matrix
  H <- 1/m*(t(X)%*%S%*%X + lambda*diag(3))
  grad <- 1/m*(t(X)%*%(y_hat - y) + lambda*theta) #gradient of J(theta) (i.e. derivative of log-likelihood)
  
  theta <- theta - solve(H)%*%grad # take step towards theta
}

theta