library(ggplot2)
library(magrittr)
library(Rtsne)
library(text2vec)
library(tm)

categories <- c("animal", "human", "supernatural", "trickster")
corpora <- lapply(categories, function(category) {
  VCorpus(DirSource(directory = file.path("corpus", category)))
})
names(corpora) <- categories
sizes <- sapply(corpora, length)
corpus <- do.call(c, corpora)

# These lines utilize the tools in the tm package for preprocessing the texts
# in the acq and the crude categories. tm_map is a function that applies a certain
# transformation on the corpus. The notation %>% can be interpreted as a forward arrow
# sending the result of one transformation to the next transformation.
corpus %<>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind = "en")) %>%
  tm_map(stripWhitespace)

# This function is used to concatenate the lines of a document in the corpus
concatenate <- function(doc) {
  paste(doc$content, collapse = '')
}

# The corpus is converted to a data frame and then into a vector so that it
# can be readily used by the GloVe algorithm in the text2vec package
m <- data.frame(text = unlist(lapply(corpus, concatenate)), stringsAsFactors = F)
m <- as.vector(m)
tokens <- space_tokenizer(m)
it = itoken(tokens, progressbar = F)
# The vocabulary is reduced to include only those terms that appear 500 times or
# more in the corpus. This is practical to do in order to reduce the complexity
# of the term map
vocab <- create_vocabulary(it) %>% prune_vocabulary(term_count_min = 500L)
# The terms are embedded into an n-dimensional (in this case n = 100) vector space
# by means of the GloVe algorithm
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
glove = GlobalVectors$new(word_vectors_size = 100, vocabulary = vocab, x_max = 10)
word_vectors_main <- glove$fit_transform(tcm, n_iter = 20)
word_vectors_context <- glove$components
word_vectors <- word_vectors_main + t(word_vectors_context)
distances <- dist(word_vectors)
# We finally use the t-SNE algorithm (t-distributed stochastic neighbor embedding) to reduce the
# dimensionality of the term space from 100 to 2 dimensions, in order to make the
# terms visualizable in a diagram
tsne <- Rtsne(word_vectors)
tdf <- data.frame(x = tsne$Y[, 1], y = tsne$Y[, 2], term = vocab$term)
ggplot(tdf, aes(x = x, y = y, label = term)) + geom_text()

# We make use of the hclust function (available in the base installation of R)
# to generate a hierarchical clustering of the terms, visualized as a dendrogram
#I uncommented the following 2 lines 

hc <- hclust(distances)
plot(hc, hang = -2)
