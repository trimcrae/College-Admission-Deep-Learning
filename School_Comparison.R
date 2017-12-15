#Tristan McRae
#Machine Learning and Causal Inference
#Final Project


#Prepare workspace
setwd("C:/Users/trimc/Desktop/Stanford/Spring2017/Econ/Final")
rm(list = ls())
library(glmnet)

#Load Data
filename = 'all.csv'
admission <- read.csv(filename)

#Remane Columns
colnames(admission) <-c ("num", "university", "field", "degree", "season", 
                         "decision", "contact", "date1", "date2", "gpa", 
                         "gre_v", "gre_q", "gre_w", "is_new_gre", "gre_subject", 
                         "status", "date3", "date4", "comments")

#Turn degree into number MS = 0, PhD = 1
admission$deg_num <- NA
admission <- within(admission, deg_num[degree=="PhD"] <- 1)
admission <- within(admission, deg_num[degree=="MS"] <- 0)

#Turn decision into number with Accepted = 1, Rejected = 0
admission$dec_num <- NA
admission <- within(admission, dec_num[decision=="Accepted"] <- 1)
admission <- within(admission, dec_num[decision=="Rejected"] <- 0)

#Turn is_new_gre number with True = 1, False = 0
admission$gre_num <- NA
admission <- within(admission, gre_num[is_new_gre=="True"] <- 1)
admission <- within(admission, gre_num[is_new_gre=="False"] <- 0)

#Select columns of interest
admission <- admission[, c("university", "deg_num", "dec_num", "gpa", "gre_v", 
                           "gre_q", "gre_w", "gre_num")]

#Remove NANs
admission <- admission[complete.cases(admission), ]

#Split into school1 and school2
school1 <- admission[grep("Stanford", admission$university),]
school2 <- admission[grep("Arizona State", admission$university),]

#Assign 0 to school2 and 1 to school1 and rejoin
school1$uni_num <- 1
school2$uni_num <- 0
admission_combo <- rbind(school1, school2)
admission_combo <- admission_combo[, c("uni_num", "dec_num", "deg_num", "gpa", "gre_v", 
                                       "gre_q", "gre_w", "gre_num")]

#Rename columns to fit conventions
#Outcome - decision
#Treatment - university
#Covariates - Degree, GPA and GRE scores
colnames(admission_combo) <- c("W", "Y", "deg_num", "gpa", "gre_v", 
                               "gre_q", "gre_w", "gre_num")
covariate.names <- c("deg_num", "gpa", "gre_v", 
                     "gre_q", "gre_w", "gre_num")
covariates <- admission_combo[covariate.names]

#Scale data
Y <- admission_combo[["Y"]]
W <- admission_combo[["W"]]
covariates.scaled <- scale(covariates)
admission.scaled <- data.frame(Y, W, covariates.scaled)

#Split into train and test data 75-25
set.seed(44)
smplmain <- sample(nrow(admission.scaled), round(3*nrow(admission.scaled)/4), replace=FALSE)
admission.train <- admission.scaled[smplmain,]
admission.test <- admission.scaled[-smplmain,]
y.train <- as.matrix(admission.train$Y, ncol=1)
y.test <- as.matrix(admission.test$Y, ncol=1)
y.test.df <- as.data.frame(y.test)

#Create linear equation
sumx = paste(covariate.names, collapse = " + ")  # "X1 + X2 + X3 + ..." for substitution later
linear <- paste("Y",paste("W",sumx, sep=" + "), sep=" ~ ")
linear <- as.formula(linear)


####################
# Naive Prediction #
####################
pred.naive <- y.test.df
pred.naive$pred <- 0
pred.naive$correct <- 0
colnames(pred.naive) <-c ("actual","pred", "correct")
pred.naive <- within(pred.naive, correct[pred==actual] <- 1)
accuracy.naive = sum(pred.naive$correct)/nrow(pred.naive)


##############
# Neural Net #
##############
library(neuralnet)


####################################
# Comparisons between universities #
####################################

#Split training data into universities
school2.train <- admission.train[grep(0, admission.train$W),]
school1.train <- admission.train[grep(1, admission.train$W),]

#Remove university column
school2.train <- school2.train[c("Y", "deg_num", "gpa", "gre_q", "gre_v", "gre_w", "gre_num")]
school1.train <- school1.train[c("Y", "deg_num", "gpa", "gre_q", "gre_v", "gre_w", "gre_num")]
comparison.test <- admission.test[c("Y", "deg_num", "gpa", "gre_q", "gre_v", "gre_w", "gre_num")]
comp.y.test.df <- as.data.frame(comparison.test$Y)


#################
# NN comparison #
#################
#Create linear equation
linear.comp <- paste("Y",sumx, sep=" ~ ")
linear.comp <- as.formula(linear.comp)

school1.nn <- neuralnet(linear.comp,data=school1.train,hidden=c(4),linear.output=F)
pred.nn.school1 <- compute(school1.nn,comparison.test[,2:7])
pred.nn.school1.df <- as.data.frame(pred.nn.school1)
pred.nn.school1.df <- pred.nn.school1.df["net.result"]
school1.average = mean(pred.nn.school1.df$net.result)

school2.nn <- neuralnet(linear.comp,data=school2.train,hidden=c(4),linear.output=F)
pred.nn.school2 <- compute(school2.nn,comparison.test[,2:7])
pred.nn.school2.df <- as.data.frame(pred.nn.school2)
pred.nn.school2.df <- pred.nn.school2.df["net.result"]
school2.average = mean(pred.nn.school2.df$net.result)


ATE.NN = school1.average - school2.average
school1.average
school2.average
ATE.NN
