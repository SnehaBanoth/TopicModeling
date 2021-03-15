# Import all the necessary packages
library(stm)
library(dplyr)
library(tidytext)
library(bigrquery)
library(purrr)
library(tidyr)
library(ggplot2)
library(furrr)
library(stminsights)
library(cluster)
library(mclust, quietly=TRUE)
library(sBIC)
library(psych)
library(stats)
library(ggpubr)
library(factoextra)
library(NbClust)


# give the folder path where the input files are
direct <- "/Users/snehabanoth/Desktop/"

#Import the Input: where text(game description along with critic text) is already preprocessed in python
table<-read.csv(paste0(direct,"preprocessed_with_critics.csv"), sep = '|')
table<-as.data.frame(table)

# Prepare the corpus
processed<- textProcessor(table$lemma_text,metadata=table,
                          lowercase=FALSE, removestopwords=FALSE, removenumbers=FALSE, 
                          removepunctuation=FALSE, stem=FALSE, sparselevel=0.9975, 
                          language="en",verbose=TRUE, onlycharacter= FALSE, striphtml=FALSE
                          )
out <- prepDocuments(processed$documents, processed$vocab, processed$meta)
docs <- out$documents
vocab <- out$vocab
meta <- out$meta

#------------------------Dont run step 3 (use 20 topics)--------------------------

# Finding the number of topics
print(paste("First heuristic to determine the number of topics:",round(sqrt(length(table$newid)/2),0)))
# simple Heuristic gives 91 topics -> very high number

# to find the number of topics using parallel processing
plan(multiprocess)

timea<-Sys.time()
many_models <- data_frame(K = c(10, 15, 20, 25, 30)) %>%
  mutate(topic_model = future_map(K, ~stm(documents=docs, vocab=vocab, K = ., max.em.its = 10,
                                          verbose = FALSE)))
print(Sys.time()-timea)

heldout <- make.heldout(documents=docs, vocab=vocab)

k_result <- many_models %>%
  mutate(exclusivity = map(topic_model, exclusivity),
         semantic_coherence = map(topic_model, semanticCoherence, documents=docs),
         eval_heldout = map(topic_model, eval.heldout, heldout$missing),
         residual = map(topic_model, checkResiduals, documents=docs),
         bound =  map_dbl(topic_model, function(x) max(x$convergence$bound)),
         lfact = map_dbl(topic_model, function(x) lfactorial(x$settings$dim$K)),
         lbound = bound + lfact,
         iterations = map_dbl(topic_model, function(x) length(x$convergence$bound)))

k_result %>%
  transmute(K,
            `Lower bound` = lbound,
            Residuals = map_dbl(residual, "dispersion"),
            `Semantic coherence` = map_dbl(semantic_coherence, mean),
            `Held-out likelihood` = map_dbl(eval_heldout, "expected.heldout")) %>%
  gather(Metric, Value, -K) %>%
  ggplot(aes(K, Value, color = Metric)) +
  geom_line(size = 1.5, alpha = 0.7, show.legend = FALSE) +
  facet_wrap(~Metric, scales = "free_y") +
  labs(x = "K (number of topics)",
       y = NULL,
       title = "Model diagnostics by number of topics",
       subtitle = "These diagnostics indicate that a good number of topics would be a number before 100")

k_result %>%
  select(K, exclusivity, semantic_coherence) %>%
  filter(K %in% c(10, 15, 20, 25, 30)) %>%
  unnest() %>%
  mutate(K = as.factor(K)) %>%
  ggplot(aes(semantic_coherence, exclusivity, color = K)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(x = "Semantic coherence",
       y = "Exclusivity",
       title = "Comparing exclusivity and semantic coherence"
       )

#----------------------------------( 20 TOPICS )-----------------------------------#

# Running the algorithm with 20 topics
STMfit <- stm(documents = docs, vocab = vocab, K = 20,data = meta, verbose=TRUE,
              max.em.its = 10,
              reportevery=5,
              prevalence = ~ Genre ,
              seed=123456789, init.type = "Random"
              )

# Topic Exploration
plot(STMfit, n=5,labeltype = "frex", topics = 1:20, type="summary")

#Sigma: covariance matrix. 
correlations <- topicCorr(STMfit,method='huge')$cor
plot(topicCorr(STMfit,method='huge'))

#create a correlation matrix to perform clustering
m <- matrix(correlations@x, ncol = 20, nrow = 20)

#We can add the Theta matrix into the original data. We can estimate new variables based on the theta values. 
STM_DATA<-cbind(meta,STMfit$theta)

#save this file for calculating the spanningness
saveRDS(STM_DATA,paste0(direct,"STM_Data_with_critics.rds"))


#----------------------------------< CLUSTERING >--------------------------------------#

#tranform into dissimilarity matrix
m2 <- (1-m)/2 

# Elbow method
fviz_nbclust(m2, kmeans, method = "wss") +
  geom_vline(xintercept = 5, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(m2, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

#fit C-means fuzzy clustering with k = 5
clustering <- fanny(m2, k=3, memb.exp = 1.2)
round(clustering$membership, 2)[1:15,]

#plot the clusters
fviz_cluster(clustering, ellipse.type = "norm")


#-----------------------------< Calculate Spanningness >-------------------------------#

#load stm data, previously saved, with topic correlations
DATA_STM <- readRDS("/Users/snehabanoth/Desktop/STM_Data_with_critics.rds")

correlations <- matrix(nrow = 20, ncol = 20)
for (i in 1:20){
  for (j in 1:20){
    chari <- as.character(i)
    charj <- as.character(j)
    correlations[i,j] <- cor(DATA_STM[[chari]], DATA_STM[[charj]])
  }  
}

distances <- 1/((correlations+1)/2)-1
distances_2 <- (1/((correlations+1)/2)-1)**4

centroids <- function(x){
  return(unlist(x)%*%distances)
}

get_distance <- function(centroid, topics){
  value <- 0
  for (i in 1:20){
    value <- value + as.numeric(dist(rbind(centroid,distances_2[i,])))*topics[i]
  }
  return(value)
}

STM_TOPICS <- DATA_STM[,c(1,6:25)]

STM_CENTROIDS <- t(apply(STM_TOPICS[,-1], 1, centroids))

spanningfactor <- c()

for (i in 1:dim(STM_CENTROIDS)[1]){
  spanningfactor <- c(spanningfactor, get_distance(STM_CENTROIDS[i,],STM_TOPICS[i,-1]))
}

df <- DATA_STM[,1:4]
spandf <- data.frame(spanning = as.numeric(spanningfactor))
spanningframe = cbind(df, spandf)

# export the results to the same folder
write.csv(spanningframe, paste0(direct,"game_spanningness.csv"))
