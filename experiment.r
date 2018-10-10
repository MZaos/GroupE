library(caret)
library(dplyr)
library(e1071)
library(magrittr)
library(randomForest)
library(stringr)
library(tm)

categories <- c("animal", "human", "supernatural", "trickster")
corpora <- lapply(categories, function(category) {
  VCorpus(DirSource(directory = file.path("corpus", category)))
})
names(corpora) <- categories
sizes <- sapply(corpora, length)
corpus <- do.call(c, corpora)

corpus %<>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind = "en")) %>%
  tm_map(stripWhitespace)

# We filter out rare terms (appearing in less than 10% of the documents)
# and generate a document-term matrix, weighted by tf-idf.
# The DT matrix is converted to a data frame and downsampled to match
# the most infrequent class.
sparsity <- 0.7
options <- list(weighting = weightTfIdf)
matrix <- DocumentTermMatrix(corpus, control = options) %>% removeSparseTerms(sparsity)
dtm <- as.data.frame(as.matrix(matrix))
dtm$Class <- rep(categories, sizes) %>% as.factor
dtm <- downSample(x = dtm[, -ncol(dtm)], y = dtm$Class)

# We reduce the vocabulary to a size of at most 30 words, using a univariate filtering method.
# See https://topepo.github.io/caret/feature-selection-using-univariate-filters.html
# for more information about this approach.
num.terms <- 30
outcome.name <- "Class"
control <- sbfControl(functions = rfSBF, method = "repeatedcv", repeats = 5)
predictors <- names(dtm)[!names(dtm) %in% outcome.name]
features <- sbf(dtm[,predictors], dtm[,outcome.name], sbfControl = control)
terms <- intersect(colnames(matrix), features$variables$selectedVars)
matrix <- matrix[, terms[1:min(length(terms), num.terms)]]
df.base <- as.data.frame(as.matrix(matrix), row.names = NULL)
df.base$Class <- rep(categories, sizes) %>% as.factor

# A baseline model is trained and the class probabilities for all the
# categories are computed. To avoid underflow, we convert the probabilities
# to the corresponding self-information values, i.e. -log(P(X)).
model.base <- naiveBayes(Class ~ ., data = df.base, laplace = 1)
prediction <- as.data.frame(-log(predict(model.base, df.base, type = "raw")))

# We generate a new data frame that will contain columns for the class probabilities
# for the categories animal, human, and supernatural
df.extd <- df.base
df.extd$animal <- prediction$animal
df.extd$human <- prediction$human
df.extd$supernatural <- prediction$supernatural
model.extd <- naiveBayes(Class ~ ., data = df.extd, laplace = 1)

# We prepare a data frame to contain the evaluation result
result <- data.frame(predicted = character(), actual = character(), model = character())
numExperiments <- 30

for (i in 1:numExperiments) {
  print(i)#partition assignated to training changed from 0.5 to 0.7
  index <- createDataPartition(df.base$Class, p = 0.7, list = F)
  
  # Split the data into a training set and a test set for the baseline model
  trainSet <- df.base[index, ]
  testSet  <- df.base[-index, ]
  
  prediction <- predict(model.base, testSet, type = "class")
  result <- rbind(result, data.frame(predicted = prediction, actual = testSet$Class, model = "baseline"))
  
  # Split the data into a training set and a test set for the extended model
  trainSet <- df.extd[index, ]
  testSet  <- df.extd[-index, ]
  
  prediction <- predict(model.extd, testSet, type = "class")
  result <- rbind(result, data.frame(predicted = prediction, actual = testSet$Class, model = "extended"))
}

result.baseline <- result %>% filter(model == "baseline")
result.extended <- result %>% filter(model == "extended")
print("Confusion matrix for the baseline strategy")
confusionMatrix(factor(result.baseline$predicted),
                factor(result.baseline$actual),
                positive = "trickster")
print("Confusion matrix for the extended strategy")
confusionMatrix(factor(result.extended$predicted),
                factor(result.extended$actual),
                positive = "trickster")