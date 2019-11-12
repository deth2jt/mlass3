
#http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/118-principal-component-analysis-in-r-prcomp-vs-princomp/
install.packages("Hmisc")

#[1] 3609  563


library(factoextra)
#PCA^^^^

foo <- read.csv("/Users/jthompson/Documents/ml/ass3/train.csv", stringsAsFactors = TRUE)
decathlon2.active <- foo[1:3609, 3:563]

res.pca <- prcomp(decathlon2.active, scale = TRUE)
#Visualize eigenvalues (scree plot). Show the percentage of variances explained by each principal component.
fviz_eig(res.pca)

# Helper function 
#::::::::::::::::::::::::::::::::::::::::
var_coord_func <- function(loadings, comp.sdev){
  loadings*comp.sdev
}
# Compute Coordinates
#::::::::::::::::::::::::::::::::::::::::
loadings <- res.pca$rotation
sdev <- res.pca$sdev
var.coord <- t(apply(loadings, 1, var_coord_func, sdev)) 
head(var.coord[, 1:4])

# Compute Cos2
#::::::::::::::::::::::::::::::::::::::::
var.cos2 <- var.coord^2
head(var.cos2[, 1:4])


# Compute contributions
#::::::::::::::::::::::::::::::::::::::::
comp.cos2 <- apply(var.cos2, 2, sum)
contrib <- function(var.cos2, comp.cos2){var.cos2*100/comp.cos2}
var.contrib <- t(apply(var.cos2,1, contrib, comp.cos2))
head(var.contrib[, 1:4])


fviz_eig(res.pca
a<-var.contrib[order(var.contrib[,1]),1]
a[540:561]

#######################MINST######################################
#https://gist.github.com/bdewilde/3965255
foo1 <- read.csv("/Users/jthompson/Documents/ml/ass3/train_m.csv", stringsAsFactors = TRUE)
decathlon.active <- foo1[1:28001, 0:784]

install.packages("caret")
install.packages("imputeTS")

library(caret)
library(imputeTS)

badCols <- nearZeroVar(decathlon.active)
decathlon.active <- decathlon.active[, -badCols]
decathlon.active<-na_mean(decathlon.active)

res1.pca <- prcomp(decathlon.active, scale = TRUE)
#Visualize eigenvalues (scree plot). Show the percentage of variances explained by each principal component.
fviz_eig(res1.pca)


loadings1 <- res1.pca$rotation
sdev1 <- res1.pca$sdev
var1.coord <- t(apply(loadings1, 1, var_coord_func, sdev1)) 
head(var1.coord[, 1:4])

# Compute Cos2
#::::::::::::::::::::::::::::::::::::::::
var1.cos2 <- var1.coord^2
head(var1.cos2[, 1:4])


# Compute contributions
#::::::::::::::::::::::::::::::::::::::::
comp1.cos2 <- apply(var1.cos2, 2, sum)
contrib1 <- function(var1.cos2, comp1.cos2){var1.cos2*100/comp1.cos2}
var1.contrib <- t(apply(var1.cos2,1, contrib1, comp1.cos2))
head(var1.contrib[, 1:4])


fviz_eig(res1.pca)
a<-var1.contrib[order(var1.contrib[,1]),1]
a[240:248]
