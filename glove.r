library(ggplot2)
library(magrittr)
library(Rtsne)
library(text2vec)
library(tm)

# We begin by loading the three categories acq, crude, and earn from the Reuters-21578 dataset.
acq <- VCorpus(DirSource("reuters/acq"))
crude <- VCorpus(DirSource("reuters/crude"))
earn <- VCorpus(DirSource("reuters/earn"))

# These lines generate a vector of the names of the categories, together with
# a list of the text corpora corresponding to these categories.
# We also generate a vector of the sizes (number of documents) in each category.
corpus.names <- c("acq", "crude", "earn")
corpus.list <- list(acq, crude, earn)
corpus.sizes <- sapply(corpus.list, length)
corpus <- do.call(c, corpus.list)

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
# The vocabulary is reduced to include only those terms that appear 30 times or
# more in the corpus. This is practical to do in order to reduce the complexity
# of the term map
vocab <- create_vocabulary(it) %>% prune_vocabulary(term_count_min = 30L)
# The terms are embedded into an n-dimensional (in this case n = 100) vector space
# by means of the GloVe algorithm
vectorizer <- vocab_vectorizer(vocab)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)
glove = GlobalVectors$new(word_vectors_size = 100, vocabulary = vocab, x_max = 10)
word_vectors_main <- glove$fit_transform(tcm, n_iter = 20)
word_vectors_context <- glove$components
word_vectors = word_vectors_main + t(word_vectors_context)
# We finally use the t-SNE algorithm (t-stochastic neighbor embedding) to reduce the
# dimensionality of the term space from 100 to 2 dimensions, in order to make the
# terms visualizable in a diagram
tsne <- Rtsne(word_vectors)
tdf <- data.frame(x = tsne$Y[, 1], y = tsne$Y[, 2], term = vocab$term)
ggplot(tdf, aes(x = x, y = y, label = term)) + geom_text()
