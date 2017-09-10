## The k conditional nearest neighbor approach (kcnn)
## This function is for the kcnn classification.
## The output of the function is a list that includes 
## the predicted classes for the test data and
## the estimated probability for each class.

# IMPORTANT:
# The train and test data are either matrix or data frame.
# The kCNN function uses the first column of the data object
# as the class variable.


kCNN <- function(train, test, k = 1, r = NULL, eps = 0.0000001, ensemble = TRUE)
{
	train <- as.matrix(train)
	test <- as.matrix(test)
	freq_label <- table(train[,1])
	n_class <- length(table(train[,1]))
	label <- names(table(train[,1]))
	X_tr <- as.matrix(train[,-1])
	X_ts <- as.matrix(test[,-1])
	X_tr2 <- list(1)
	for(j in 1:n_class)
	{
  		ind <- which(train[,1]==label[j])
  		X_tr2[[j]] <- X_tr[ind,]
	}
	p <- ncol(X_tr)
	if(is.null(r)) r <- p

	predict_kcnn <- numeric(nrow(test))
	prob_kcnn <- matrix(0,nrow=nrow(test),ncol=(n_class))
	result2 <- numeric(n_class)

	for(i in 1:nrow(test))
	{
      result <- prob <- matrix(0,nrow=k,ncol=(n_class))
  		for(j in 1:n_class)
  		{
    		if(freq_label[j]>1) temp2 <- t(t(X_tr2[[j]])-X_ts[i,])
    		if(freq_label[j]>1) dx <- sqrt(apply(temp2*temp2,1,sum)) + eps
    		if(freq_label[j]==1) temp2 <- (X_tr2[[j]]-X_ts[i,])
    		if(freq_label[j]==1) dx <- sqrt(sum((temp2*temp2))) + eps
    		result[,j] <- sort(dx)[1:k]
    		ind <- order(dx)[1:k]    
    		if(length(dx)==1) temp <- X_tr2[[j]]
    		if(length(dx)>1)
    		{
      			if(k==1) temp <- X_tr2[[j]][ind,]
      			if(k>1 && k<length(dx)) temp <- apply(X_tr2[[j]][ind,],2,mean)    
      			if(k>1 && k>=length(dx)) temp <- apply(X_tr2[[j]],2,mean)
    		}
      		result2[j] <- sqrt(sum((temp-X_ts[i,])^2))
  		}

  		if(ensemble=="TRUE")
  		{
			  for(j in 1:k)
  			{
    			denom <- sum(k/result[j,]^(p/r),na.rm=TRUE)
          if(denom>0)	prob[j,] <- (k/result[j,]^(p/r))/denom
          if(denom==0) prob[j,which.min(result[j,])] <- 1
    			ind <- which(is.na(prob[j,]))
    			prob[j,ind] <- 1
  			}
  			vote_prob <- prob_kcnn[i,] <- apply(prob,2,mean,na.rm=TRUE)
  			if(sum(max(vote_prob)==vote_prob)==1)
  			{
    			predict_kcnn[i] <- as.numeric(label[which.max(vote_prob)])
  			}
  			if(sum(max(vote_prob)==vote_prob)>1)
  			{
    			ind <- which(max(vote_prob)==vote_prob)
    			a <- sample(ind,1)
    			predict_kcnn[i] <- as.numeric(label[a])
  			}
  		}

  		if(ensemble=="FALSE")
  		{
          denom <- sum(k/result[k,]^(p/r),na.rm=TRUE)
          if(denom>0) prob_kcnn[i,] <- (k/result[k,]^(p/r))/denom
          if(denom==0) prob_kcnn[i,which.min(result[k,])] <- 1
  			  ind <- which(is.na(prob_kcnn[i,]))
  			  prob_kcnn[i,ind] <- 1
    		if(length(ind)==1) prob_kcnn[i,ind] <- 1
    		if(length(ind)>1) 
    		{
      			prob_kcnn[i,] <- rep(0,n_class)     
      			ind2 <- which.max(k/result[k,])
      			prob_kcnn[i,ind2] <- 1
    		}
		  	vote_prob <- prob_kcnn[i,]
  			if(sum(max(vote_prob,na.rm=TRUE)==vote_prob,na.rm=TRUE)==1)
  			{
    			predict_kcnn[i] <- as.numeric(label[which.max(vote_prob)])
  			}
  			if(sum(max(vote_prob,na.rm=TRUE)==vote_prob,na.rm=TRUE)>1)
  			{
    			ind <- which(max(vote_prob,na.rm=TRUE)==vote_prob)
    			a <- sample(ind,1)
    			predict_kcnn[i] <- as.numeric(label[a])
  			}
  		}
	}	
	if(ensemble==FALSE) return(list(predict_kcnn=predict_kcnn,probability_kcnn=prob_kcnn))
  if(ensemble==TRUE) return(list(predict_ekcnn=predict_kcnn,probability_ekcnn=prob_kcnn))
}


