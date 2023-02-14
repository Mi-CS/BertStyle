# Datasets

## Gutenberg

The R package [gutenbergr](https://cran.r-project.org/web/packages/gutenbergr/vignettes/intro.html) was employed for selecting the subset and downloading the text files. It was first filtered by books written only in English, and then by the 50 authors that had more books available for download. See `gutenberg_data_retrieval.R`. 

To build the sample employed as example, we chose Charles Dickens, Nathaniel Hawthorne, and George Alfred Henty. 

## Reuters 

Obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Reuter_50_50). We merged these, and sampled 80% of the articles for each author. 

For the example, we chose Benjamin Kang Lim, Samuel Perry, and William Kazer. The dataset is originally split 50% into train and test sets. 

## ArXiv cs.CL preprints

The dataset has been built employing this [arXiv wrapper](https://github.com/Mi-CS/simple_arXiv). We searched for the manuscripts in the ``cs.CL`` category, and download them as `pdf`. As parser, [Apache Tika](https://tika.apache.org) was employed. Metadata on the manuscripts and the download link can be found in `metadata_papers_csCL.json`. 

For the current example, all 2021 manuscripts from Diptesh Kanojia, Hannes Westermann, and Bing Liu were chosen. 

*Note that the cs.CL dataset must be re-built to run the examples presented in this repository.*