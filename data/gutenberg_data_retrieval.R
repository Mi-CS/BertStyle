### R script to download Gutenberg books using
### gutenbergr package


# Required libraries
library(gutenbergr)
library(dplyr)

# Retrieve metadata on books. This creates a list
# With the following fields:
# ("gutenberg_id", "title", "author", "gutenberg_author_id",
# "language", "gutenberg_bookshelf", "rights", "has_text")
gut_en <- gutenberg_works(languages = "en")

# Filter out entries with the following authors
del_auth <- c("Various", NA, "Anonymous", "Unknown",
 "Library of Congress. Copyright Office")

# Get 50 authors with the largest number of written books
authors_subset <- gut_en %>%
  group_by(author) %>%
  summarise(n = n()) %>%
  arrange(-n) %>%
  filter(!author %in% del_auth) %>%
  top_n(50, wt = n) %>%
  pull(author)


# Get meta-data for 50-authors subset
subset_metadata <- gutenberg_works(languages = "en") %>%
  filter(author %in% authors_subset)

# Get subset ids
gut_ids <- subset_metadata %>%
  pull(gutenberg_id)

# Metadata to .csv file
write.csv(subset_metadata, file = "metadata.csv")

# Loop over subset of books and save them as .txt
# The folder books in the working directory must exist.
for (book_id in gut_ids) {
  zip_id <- paste(book_id, ".zip", sep = "")
  df.text <- gutenberg_download(book_id, mirror = "http://aleph.gutenberg.org")
  cat(df.text$text,
      file = paste("books/", book_id, ".txt", sep = ""),
      sep = " ")
}