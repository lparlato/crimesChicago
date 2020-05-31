## installing packages (Not using if(!require()) since I am using for sure all of them)
install.packages("data.table") ## manipulation
install.packages("ggplot2") ## visualization
install.packages("GGally") ## Descriptive Visualization
install.packages("plyr") ## aggregation
install.packages("stringr") ## string manipulation
install.packages("glmnet") ## Cross-Validation, Ridge and Lasso Regression
install.packages("lubridate") ## working with dates
install.packages("xtable") ## latex tables
install.packages("caret") ## confusion Matrix 
install.packages("e1071") ## dependency of confusion matrix

## loading packages
library(data.table)
library(ggplot2)
library(GGally)
library(plyr)
library(stringr)
library(glmnet)
library(lubridate)
library(xtable)
library(caret)
library(e1071)

## loading data
crimeData <- fread(unzip("crimes.zip"))
file.remove("Crimes_-_2001_to_present.csv")

## Visualizing top 5 observations
xtable(crimeData[1:5,1:7])
xtable(crimeData[1:5,8:15])
xtable(crimeData[1:5,16:22])

set.seed(1234)
## adapting date
crimeData$Date <- as.Date(crimeData$Date, format = "%m/%d/%Y %I:%M:%S %p")
crimeData$year <- year(crimeData$Date)
crimeData$month <- month(crimeData$Date)
crimeData$day <- day(crimeData$Date)

## Arrest, Domestic, IUCR and FBI Code as factor
crimeData$Arrest <- as.factor(crimeData$Arrest)
crimeData$Domestic <- as.factor(crimeData$Domestic)
crimeData$IUCR <- as.factor(crimeData$IUCR)
crimeData$`FBI Code` <- as.factor(crimeData$`FBI Code`)

## training set, tunning set and test set (80% training, 10% tunning, 10% testing)
obs_out <- sample(1:nrow(crimeData), round(nrow(crimeData)*0.2))

obs_out_tunning <- obs_out[1:(length(obs_out)/2)]
obs_out_testing <- obs_out[((length(obs_out)/2) + 1):length(obs_out)]

training <- crimeData[!obs_out,]
testing <- crimeData[obs_out_testing,]
tuning <- crimeData[obs_out_tunning,]

############################
## Descriptive Statistics ##
############################

str(training) ## structure of variables
xtable(summary(training)) ## summary of values

## Removing variables with NAs and not going to be used
training <- training[,-c(13,14,16:22)]
testing <- testing[,-c(13,14,16:22)]
tuning <- tuning[,-c(13,14,16:22)]

## drop id, case number, (primary type and description - using IUCR)
training <- training[,-c(1,2,6,7)] 
testing <- testing[,-c(1,2,6,7)]
tuning <- tuning[,-c(1,2,6,7)]

## drop block and location description since usinb Beat and District
training <- training[,-c(1,2,4)]
testing <- testing[,-c(1,2,4)]
tuning <- tuning[,-c(1,2,4)]

## handling NA observations for Disctrit
training$District[is.na(training$District)] <- -1
testing$District[is.na(testing$District)] <- -1
tuning$District[is.na(tuning$District)] <- -1

## visualizing correlations 
ggpairs(training[,-c(1,6)])

## Visualizing IUCR
ggplot(training) + geom_bar(aes(factor(IUCR))) +
  theme(axis.text.x = element_text(angle = 90)) 

## Visualizing FBI Code 
ggplot(training) + geom_bar(aes(factor(`FBI Code`)))

####################
## Modeling Data ###
####################

## create second train and test set based on training
set.seed(4321)
obs_out2 <- sample(1:nrow(training), round(nrow(training)*0.3))

train <- training[!obs_out2,]
test <- training[obs_out2,]

## separating label from features
cat_labels <- train[,2]
features <- train[,-2]

## function for adding legend to fit
lbs_fun <- function(fit, ...) {
  L <- length(fit$lambda)
  x <- log(fit$lambda[L])
  y <- fit$beta[, L]
  labs <- names(y)
  legend('bottomright', legend=labs, col=1:length(labs), lty=1) # <<< ADDED BY ME
}


## training binomial logistic regression for arrests
cvfit <- glmnet(data.matrix(features),
                cat_labels$Arrest,
                family = "binomial", trace.it = 1)

plot(cvfit, xvar = "lambda")
lbs_fun(cvfit)

coef(cvfit, s = 0.01)

predict_min_lambda <- predict(cvfit, newx = data.matrix(test[,-2]), s= 0.01, type = "class")

### calculation on F-measure for the prediction
cm_regular <-confusionMatrix(as.factor(predict_min_lambda), reference = as.factor(test$Arrest)) 
f_byclass_regular <- cm_regular[["byClass"]][["F1"]]

cm_regular ## checking how good was the model using whole train set
f_byclass_regular ## micro f-measure

############### Cross-Validating with 10-Fold. Why 10? To have enough data on each training
############### Checking if Ridge or Lasso could improve precision


## training binomial logistic regression ridge with cross-validation 10-fold
cvfit10 <- cv.glmnet(data.matrix(features),
                     cat_labels$Arrest,
                     nfolds = 10,
                     alpha = 0,
                     family = "binomial", trace.it = 1)

plot(cvfit10)

coef(cvfit10, s = "lambda.min")

predict_min_lambda <- predict(cvfit10, newx = data.matrix(test[,-2]), s= "lambda.min", type = "class")
cm_ridge <- confusionMatrix(as.factor(predict_min_lambda), reference = as.factor(test$Arrest))
f_byclass_ridge <- cm_ridge[["byClass"]][["F1"]]

cm_ridge ## checking model for under or overfitting
f_byclass_ridge ## micro f-measure

## training multinomial logistic regression lasso with cross-validation 10-fold
cvfit10_lasso <- cv.glmnet(data.matrix(features),
                           cat_labels$Arrest,
                           nfolds = 10,
                           alpha = 1,
                           family = "binomial", trace.it = 1)

plot(cvfit10_lasso)

coef(cvfit10_lasso, s = "lambda.min")

predict_min_lambda <- predict(cvfit10_lasso, newx =data.matrix(test[,-2]), s= "lambda.min", type = "class")
cm_lasso <- confusionMatrix(as.factor(predict_min_lambda), reference = as.factor(test$Arrest))
f_byclass_lasso <- cm_lasso[["byClass"]][["F1"]] 

cm_lasso ## checking model for under or overfitting
f_byclass_lasso ## micro f-measure

########
rbind(f_byclass_regular,f_byclass_ridge,f_byclass_lasso)

#####################################
##### Parcial Points:
##
## 1- The result shows an improvement when using ridge or lasso 'selection' instead of using the regular one. 
## 2- Ridge and Lasso perfomerd well.
## 3- Among Ridge and Lasso, Laso was slightly better.
##    From the above, the chosen model was using lasso and lambda = 0.001141191 (cvfit10_lasso$lambda.min)

#########################
#### 10-fold all data ###
#########################

data2 <- training ## backup

set.seed(54321)
shuffle_data <- data2[sample(nrow(data2)),] ## shuffleling data to create random folds

size9 <- ceiling(nrow(shuffle_data)/10) ## size of each fold
size10 <- nrow(shuffle_data) - size9*9
shuffle_data$fold <- c(rep(1:9, each = size9),rep(10,size10))

beta_i <- list()
f_byclass <- list()

for(i in 1:10){
  train_dummy <- shuffle_data[shuffle_data$fold != i,]
  test_dummy <- shuffle_data[shuffle_data$fold == i,]
  
  fit_dummy <- glmnet(data.matrix(train_dummy[,-c(2,10)]),
                      train_dummy$Arrest,
                      alpha = 1,
                      lambda = cvfit10_lasso$lambda.min,
                      family = "binomial", trace.it = 1)
  
  beta_i[[i]] <- coef(fit_dummy, s = "lambda.min")
  
  predict_dummy <- predict(fit_dummy, newx = data.matrix(test_dummy[,-c(2,10)]), type = "class")
  cm_dummy <- confusionMatrix(as.factor(predict_dummy), reference = as.factor(test_dummy$Arrest))
  f_byclass_dummy <- cm_dummy[["byClass"]][["F1"]] 
  
  f_byclass[[i]] <- f_byclass_dummy
}

#as.data.frame(beta_i[[1]]$`1`)

f_data <- t(as.data.frame(f_byclass))
f_mean <- colMeans(f_data, na.rm = T)

f_mean


beta_mean <- data.frame(matrix(ncol = 10, nrow = 9))

for(i in 1:10){
    list_dummy <- as.matrix(beta_i[[i]])
    beta_mean[i] <- data.matrix(list_dummy)
}

## means beta
beta_final <- as.matrix(beta_mean) %>% apply(1,mean)
beta_final

## variation on beta_final
var_beta_final <- as.matrix(beta_mean) %>% apply(1,var)
var_beta_final

data.frame(beta_final, var_beta_final)


##########################################################################
## Predicting on tuning set using beta_final for tuning threshold class ##
##########################################################################

predict_tuning <- beta_final[1] + beta_final[2]*as.numeric(tuning$IUCR) +
                                 beta_final[3]*as.numeric(tuning$Domestic) +
                                 beta_final[4]*tuning$Beat +
                                 beta_final[5]*tuning$District +
                                 beta_final[6]*as.numeric(tuning$`FBI Code`) +
                                 beta_final[7]*tuning$year +
                                 beta_final[8]*tuning$month +
                                 beta_final[9]*tuning$day

summary(predict_tuning) 

## Finding proportional of true/false on training set
data_prop <- count(training$Arrest) 
data_prop2 <- count(tuning$Arrest)
predict_tuning <- as.data.frame(predict_tuning)

quantile(predict_tuning$predict_tuning, data_prop$freq[data_prop$x == FALSE] / sum(data_prop$freq))
quantile(predict_tuning$predict_tuning, data_prop2$freq[data_prop2$x == FALSE] / sum(data_prop2$freq))

## testing threshold = median
predict_tuning <- as.data.frame(predict_tuning)
predict_tuning$true <- FALSE
predict_tuning$true[predict_tuning$predict_tuning >= -1.4856] <- TRUE

cm_tuning_median <- confusionMatrix(as.factor(predict_tuning$true), reference = as.factor(testing$Arrest))
f_byclass_tuning_median <- cm_tuning_median[["byClass"]][["F1"]] 

## testing threshold = mean
predict_tuning$true <- FALSE
predict_tuning$true[predict_tuning$predict_tuning >= -1.104] <- TRUE

cm_tuning_mean <- confusionMatrix(as.factor(predict_tuning$true), reference = as.factor(testing$Arrest))
f_byclass_tuning_mean <- cm_tuning_mean[["byClass"]][["F1"]] 


## testing threshold = -0.51 given by proprotion 
predict_tuning$true <- FALSE
predict_tuning$true[predict_tuning$predict_tuning >= -0.51] <- TRUE

cm_tuning_51 <- confusionMatrix(as.factor(predict_tuning$true), reference = as.factor(testing$Arrest))
f_byclass_tuning_51 <- cm_tuning_51[["byClass"]][["F1"]] 

## testing threshold = -0.44 (3rd Qu) (approx. prop TRUE/FALSE in training$Arrest and tuning$Arrest 
predict_tuning$true <- FALSE
predict_tuning$true[predict_tuning$predict_tuning >= -0.44] <- TRUE

cm_tuning_44 <- confusionMatrix(as.factor(predict_tuning$true), reference = as.factor(testing$Arrest))
f_byclass_tuning_44 <- cm_tuning_44[["byClass"]][["F1"]] 

xtable(cm_tuning_median$table)
xtable(cm_tuning_mean$table)
xtable(cm_tuning_51$table)
xtable(cm_tuning_44$table)
xtable(rbind(f_byclass_tuning_median, f_byclass_tuning_mean, f_byclass_tuning_51, f_byclass_tuning_44))

#######################################################################
## Predicting on testing set using beta_final with threshold = -0.44 ##
#######################################################################

predict_final <- beta_final[1] + beta_final[2]*as.numeric(testing$IUCR) +
                                 beta_final[3]*as.numeric(testing$Domestic) +
                                 beta_final[4]*testing$Beat +
                                 beta_final[5]*testing$District +
                                 beta_final[6]*as.numeric(testing$`FBI Code`) +
                                 beta_final[7]*testing$year +
                                 beta_final[8]*testing$month +
                                 beta_final[9]*testing$day

## testing threshold = -0.44
predict_final <- as.data.frame(predict_final)
predict_final$true <- FALSE
predict_final$true[predict_final$predict_final >= -0.51] <- TRUE

cm_final<- confusionMatrix(as.factor(predict_final$true), reference = as.factor(testing$Arrest))
f_byclass_final <- cm_final[["byClass"]][["F1"]] 

## testing cvfit10 on testing
predict_min_lambda <- predict(cvfit10, newx =data.matrix(testing[,-2]), s= 'lambda.min', type = "class")
cm2 <- confusionMatrix(as.factor(predict_min_lambda), reference = as.factor(testing$Arrest))
f_byclass_ridge2 <- cm2[["byClass"]][["F1"]] 

f_byclass_ridge2 ## f-measure


## testing cvfit10_lasso on testing
predict_min_lambda <- predict(cvfit10_lasso, newx =data.matrix(testing[,-2]),  s= 'lambda.min', type = "class")
cm_lasso2 <- confusionMatrix(as.factor(predict_min_lambda), reference = as.factor(testing$Arrest))
f_byclass_lasso2 <- cm_lasso2[["byClass"]][["F1"]] 

f_byclass_lasso2 ##  f-measure

## testing cvfit on testing
predict_min_lambda <- predict(cvfit, newx =data.matrix(testing[,-2]), s = 0.01, type = "class")
cm1 <- confusionMatrix(as.factor(predict_min_lambda), reference = as.factor(testing$Arrest))
f_byclass_1 <- cm1[["byClass"]][["F1"]] 

f_byclass_1 ## f-measure

## Overall Performance before and after splitting manually on testing set
rbind(f_byclass_final, f_byclass_lasso2, f_byclass_ridge2, f_byclass_1)

