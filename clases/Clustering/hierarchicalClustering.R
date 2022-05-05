# Referencias
# https://www.statology.org/hierarchical-clustering-in-r/
# https://online.stat.psu.edu/stat505/lesson/14/14.7
# https://www.datacamp.com/community/tutorials/pca-analysis-r
# https://www.guru99.com/r-scatter-plot-ggplot2.html

library(factoextra)
library(cluster)

#load data
df <- USArrests
#remove rows with missing values
df <- na.omit(df)
#scale each variable to have a mean of 0 and sd of 1
df <- scale(df)
#view first six rows of dataset
head(df)
# define linkage methods
m <- c("average", "single", "complete", "ward")
names(m) <- c("average", "single", "complete", "ward")

ac <- function(x) {
  agnes(df, method = x)$ac
}

#calculate agglomerative coefficient for each clustering linkage method
sapply(m, ac)

#perform hierarchical clustering using Ward's minimum variance
clust <- agnes(df, method = "ward")

#produce dendrogram
pltree(clust, cex = 0.6, hang = -1, main = "Dendrogram") 

#calculate gap statistic for each number of clusters (up to 10 clusters)
gap_stat <- clusGap(df, FUN = hcut, nstart = 25, K.max = 10, B = 50)

#produce plot of clusters vs. gap statistic
fviz_gap_stat(gap_stat)

#compute distance matrix
d <- dist(df, method = "euclidean")

#perform hierarchical clustering using Ward's method
final_clust <- hclust(d, method = "ward.D2" )

#cut the dendrogram into 4 clusters
groups <- cutree(final_clust, k=4)

final_data <- cbind(USArrests, cluster = groups)

df.pca <- prcomp(df, center = FALSE,scale. = FALSE)
summary(df.pca)

pcadf <- data.frame(df.pca$x)

# Graficar con plot
plot(pcadf[,1], pcadf[,2], main="PCA USArrests", xlab="PC1", ylab="PC2",
     col=groups, pch=16)

library("ggplot2")

# Graficar con ggplot
ggplot(pcadf, aes(x=PC1, y=PC2))+geom_point(color=factor(groups))
