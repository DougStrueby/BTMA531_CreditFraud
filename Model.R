library(data.table)
library(tree)
library(randomForest)
library(gbm)
library(ISLR)

################################reading csv################################

fraudData = fread("fraud.csv.xz")

str(fraudData)
#setting attributes to correct type
fraudData$trans_date_trans_time = as.Date(fraudData$trans_date_trans_time , format = "%d/%m/%Y")
fraudData$is_fraud = as.logical(fraudData$is_fraud)
fraudData$category = as.factor(fraudData$category)
fraudData$gender = as.factor(fraudData$gender)
fraudData$state = as.factor(fraudData$state)
fraudData$job = as.factor(fraudData$job)
fraudData$merchant = as.factor(fraudData$merchant)

#scaling
fraudData$amt = scale(fraudData$amt)

#splitting negative and postive cases
fraudDataPos = fraudData[which(fraudData$is_fraud == 1)]
fraudDataNeg = fraudData[which(fraudData$is_fraud == 0)]

#function to randomize the negative cases
#
#@param negative cases - cases where is_fraud is 0
#@param postive cases - cases where is_fraud is 1
frauDataBalanced <- function(negativeCases, positiveCases){
  fraudDataBal = rbind(
    negativeCases[sample(nrow(negativeCases), nrow(positiveCases), replace = F)],
    positiveCases
  )
  return(fraudDataBal)
}

################################classification tree model################################
cTrees = list(0)
predCTrees = list(0)
cms = list(0)
acc = numeric(0)

#loop to run model with 10 different sets of data
for(i in 1:10){
  #randomizing data
  balancedData = frauDataBalanced(fraudDataNeg, fraudDataPos)
  index = sample(nrow(balancedData), round(nrow(balancedData) * 0.8), replace = F)
  fraudTrain = balancedData[index]
  fraudTest = balancedData[-index]
  
  #running model
  cTree = tree(is_fraud ~category + gender + city_pop + amt, fraudTrain)
  #storing each tree in a list
  cTrees[[i]] = cTree
  
  #results
  plot(cTree)
  text(cTree, pretty = 0)
  
  #running prediction
  predCTree = predict(cTree, fraudTest)
  predCTree = round(predCTree)
  
  #storing each prediction in a list
  predCTrees[[i]] = predCTree
  
  #confusion matrix
  cm = table(predCTree, fraudTest$is_fraud)
  print(cm)
  
  #storing each confusion matrix in a list
  cms[[i]] = cm
  
  #accuracy
  acc[i] = sum(diag(cm))/sum(cm)
}

mean(acc)

################################randomforest################################
rForests = list(0)
predRForests = list(0)
cms = list(0)
acc = numeric(0)

#loop to run model with 10 different sets of data
for(i in 1:10){
  #randomizing data
  balancedData = frauDataBalanced(fraudDataNeg, fraudDataPos)
  index = sample(nrow(balancedData), round(nrow(balancedData) * 0.8), replace = F)
  fraudTrain = balancedData[index]
  fraudTest = balancedData[-index]
  
  #running model
  rForest = randomForest(is_fraud ~category + gender + city_pop + amt, fraudTrain, n.trees = 100)
  #storing each tree in a list
  rForests[[i]] = rForest
  
  #running prediction
  predRForest = predict(rForest, fraudTest)
  predRForest = round(predRForest)
  
  #storing each prediction in a list
  predRForests[[i]] = predRForest
  
  #confusion matrix
  cm = table(predRForest, fraudTest$is_fraud)
  print(cm)
  
  #storing each confusion matrix in a list
  cms[[i]] = cm
  
  #accuracy
  acc[i] = sum(diag(cm))/sum(cm)
}

mean(acc)

################################gbm################################
gbms = list(0)
predGbms = list(0)
cms = list(0)
acc = numeric(0)

for(i in 1:10){
  #randomizing data
  balancedData = frauDataBalanced(fraudDataNeg, fraudDataPos)
  index = sample(nrow(balancedData), round(nrow(balancedData) * 0.8), replace = F)
  fraudTrain = balancedData[index]
  fraudTest = balancedData[-index]
  
  #running model
  gbm = gbm(is_fraud ~merchant + category + gender + state + city_pop + job + amt,
            distribution = "gaussian", data = fraudTrain, n.trees = 100, 
            interaction.depth = 1, shrinkage = 0.001, 
            cv.folds = 5, n.cores = NULL, verbose = F)
  #storing each tree in a list
  gbms[[i]] = gbm
  
  #running prediction
  predGbm = predict(gbm, fraudTest)
  predGbm = round(predGbm)
  
  #storing each prediction in a list
  predGbms[[i]] = predGbm
  
  #confusion matrix
  cm = table(predGbm, fraudTest$is_fraud)
  print(cm)
  
  #storing each confusion matrix in a list
  cms[[i]] = cm
  
  #accuracy
  acc[i] = sum(diag(cm))/sum(cm)
}

mean(acc)

################################logistic regression################################
logs = list(0)
predLogs = list(0)
cms = list(0)
acc = numeric(0)

for(i in 1:10){
  #randomizing data
  balancedData = frauDataBalanced(fraudDataNeg, fraudDataPos)
  index = sample(nrow(balancedData), round(nrow(balancedData) * 0.8), replace = F)
  fraudTrain = balancedData[index]
  fraudTest = balancedData[-index]
  
  #running model
  log = glm(is_fraud ~category + gender + city_pop + amt, fraudTrain, family = "binomial")
  
  #storing each tree in a list
  logs[[i]] = log
  
  #running prediction
  predLog = predict(log, fraudTest, type = "response")
  predLog = ifelse(predLog > 0.5, 1, 0) #IDK if this is the best way to handle this
  
  #storing each prediction in a list
  predLogs[[i]] = predLog
  
  #confusion matrix
  cm = table(predLog, fraudTest$is_fraud)
  print(cm)
  
  #storing each confusion matrix in a list
  cms[[i]] = cm
  
  #accuracy
  acc[i] = sum(diag(cm))/sum(cm)
}

mean(acc)
