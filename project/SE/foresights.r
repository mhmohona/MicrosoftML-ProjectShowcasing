# Frequentist Statistician-centric Public Engagement MLSecOps Foresights
# Using [the statisticians' favourite: Microsoft's] R for [binomial] logistic regression model [imputed variables' response probability w/o ROC] foresights evaluation along with reconceptualization of a High Level Envisioned [MIT/NYT538] Methodology

# Averaging [reading in data to be aggregated scored and weighted]
Engagement=read.csv("https://github.com/Foresights-IT/MicrosoftML/EngagementData.csv")
str(Engagement)
table(Engagement$Period)
summary(Engagement)

# Adjusting [latest trends and imputations (needs must)];
set.seed(101)
# TODO: plotting the Probability Decision Surface [for Logistic Regression on a Binary Classification Task/Dataset] using DTL probabilities' foresights [with numpy, sklearn & matplotlib]

# Analysing [using multicollinearity logistical regression model with training subset predictions]
TrainSubset=subset(Engagement,Period==2012|Period==2016)
TestSubsetSubset=subset(Engagement,Period==2020)
table(TrainSubset$DependantVar0)
sign(3)
sign(-2)
sign(0)

# Snapshotting [concatenating data with analysis to produce an estimate foresight]
table(sign(TrainSubset$Survey1))
table(TrainSubset$DependantVar0,sign(TrainSubset$Survey1))

# Projecting [applying leads trends-based discount and smart model baselining accuracy]
str(TrainSubset)
cor(TrainSubset[c("Survey1","Survey2","PropVar0","Var0DiffCount","DependantVar0")])
LogRegModel1=glm(DependantVar0~PropVar0,data=TrainSubset,family="binomial")
summary(LogRegModel1)
PropPrediction1=predict(LogRegModel1,type="response")
table(TrainSubset$DependantVar0,PropPrediction1>=0.5)
LogRegModel2=glm(DependantVar0~Survey2+Var0DiffCount,data=TrainSubset,family="binomial")
PropPrediction2=predict(LogRegModel2,type="response")
table(TrainSubset$DependantVar0,PropPrediction2>=0.5)
summary(LogRegModel2)

# Simulating [probabilistic assessment end result of projected uncertain estimates]
table(TestSubset$DependantVar0,sign(TestSubset$Survey1))
TestForesights=predict(LogRegModel2,newdata=TestSubset,type="response")
table(TestSubset$DependantVar0,TestForesights>=0.5)

# Evaluating [error analysis with test subset foresights, post votes casting]
subset(TestSubset,TestForesights>=0.5&DependantVar0==0)
