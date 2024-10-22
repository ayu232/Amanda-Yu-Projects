library(dplyr)
install.packages("tidyr")
library("tidyr")


library(readr)
mydata <- read_csv("Desktop/Life Expectancy Data.csv")
df = subset(mydata, Status=='Developing')
life = subset(df, select = c(Country, Year, `Life expectancy`, `Adult Mortality`,
                               Measles, BMI, `under-five deaths`, Polio, Diphtheria,
                               `HIV/AIDS`, GDP, `Income composition of resources`,
                               `Schooling`, Alcohol))
life <- life %>% drop_na()



mlr=lm(`Life expectancy`~`Adult Mortality`+Alcohol+Measles+BMI+`under-five deaths`
       +Polio+Diphtheria+`HIV/AIDS`+GDP+`Income composition of resources`
       +Schooling, data=life)
summary(mlr)

mlr2<-step(mlr, data=life, direction = "both", test="F")
summary(mlr2)

par(mfrow=c(1,1))
plot(life$Schooling, life$`Life expectancy`, xlab="Schooling", ylab="Life Expectancy",
     main="Life Expectacy vs. Schooling", pch=16, col="blue")
abline(lm(`Life expectancy`~Schooling, data=life),col="red")
cor.test(life$Schooling, life$`Life expectancy`)

##################
edu=lm(life$`Life expectancy`~life$Schooling)
plot(life$`Life expectancy`~life$Schooling)
summary(edu)

inco= lm(life$`Life expectancy`~life$Schooling)
plot(life$`Life expectancy`~life$`Income composition of resources`)
summary(inco)

install.packages("car")
library(car)
install.packages ("gvlma")
library(gvlma)
gvlma(mlr)


influencePlot(mlr, id=list(method="identify"), main="influence plot")







