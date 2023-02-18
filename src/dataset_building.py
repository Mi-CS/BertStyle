from typing import Optional
import re
import random
import nltk
from tqdm import tqdm

NLTK_NOUNS_TAGS = ["NN", "NNS", "NNP", "NNPS"]

def _build_chunks(slist: list,
                 chunk_length: Optional[int] = 3,
                 max_sent: Optional[int] = None) -> list:
    """
    Group sentences into chunks of 'chunk_length' consecutive
    sentences. 
    
    Parameters: 
        slist (list of str): List of single sentences.
        chunk_length (int) (def. 3): Length of chunk.
        max_sent (int) (def. None): Maximum number of sentences
            to take into account for the chunks. Default None,
            which takes all sentences. 
    
    Returns: 
        chunks (list of lists): List of the resulting chunks. 
    """
    
    # Max number of sentences to take into account per document
    slist = slist[:max_sent] if max_sent else slist
    
    total_length = len(slist)
    chunks =  [slist[i:i+chunk_length] for i in 
                range(0, total_length, chunk_length)]
    
    # Remove last chunk if it is too small
    if len(chunks[-1]) != chunk_length: 
        del chunks[-1]
    
    chunks = [" ".join(chunk) for chunk in chunks]
    return chunks

def _gather_nouns(chunk_tags: list,
                 pos_tags: Optional[list] = NLTK_NOUNS_TAGS) -> list: 
    """
    Gather all the nouns in a tagged chunk employing the
    pos_tag method from the nltk package. 

    Parameters: 
        chunk_tags (list): List of tuples of the form 
            (word, tag)
        pos_tags (list): List of strings indicating which
            tags to take into account when gathering nouns. By
            default uses ["NN", "NNS", "NNP", "NNPS"]
    """
    return [word for word, tag in chunk_tags if tag in pos_tags]



def _build_dataset_from_chunks(doc_chunks: list,
                  masking_percentage: Optional[float] = 0, 
                  max_pairs_per_doc: Optional[int] = None,
                  show_progress_bar: bool = True) -> list: 
    """
    Output a list of masked positive pairs from a list of documents made of a 
    list of chunks.
    
        Parameters: 
            doc_chunks (list): list of lists, each entry representing a 
                document, and containing a list of its chunks. 
            masking_percentage (float) (def. 0): float between 0 and 1 indicating
                the percentage of noun masking to apply. By default it does not
                apply any.
            max_pairs_per_doc (int) (def. None): maximum number of positive pairs
                to produce per document. By default builds document_length // 2 
                pairs. If the number given is bigger than the default value, it
                ignores it.
            show_progress_bar (bool) (def. True): Whether to show a progress bar.

        Returns: 
            positive_pairs (list): list of positive pairs. 
    """

    assert 0 <= masking_percentage <= 1, "Masking must be a float between 0 and 1"

    positive_pairs = []
    
    disable = not show_progress_bar    
    for document in tqdm(doc_chunks, disable=disable): 
        
        # Maximum number of pairs per document
        pairs_per_doc = min(len(document)//2, max_pairs_per_doc) if\
            max_pairs_per_doc else len(document)//2

        # Randomly select the number of chunks specified
        chunks = random.sample(document, pairs_per_doc*2)

        # Mask the chunks
        if masking_percentage > 0: 
            masked_chunks = []
            for chunk in chunks:
                # Get noun tags for the chunk
                nouns = nltk.pos_tag(nltk.tokenize.word_tokenize(chunk))
                nouns = _gather_nouns(nouns)

                # Sample nouns in chunk to be masked
                nouns = random.sample(nouns,
                                     k = int(masking_percentage*len(nouns)))
                for noun in nouns:
                    regex_sub = r"([^\w])" + re.escape(noun) + r"([^\w])"
                    chunk = re.sub(regex_sub, r"\1[MASK]\2", chunk)     
                masked_chunks.append(chunk)
            
            # Update document chunks with masked ones
            chunks = masked_chunks

        # Add positive pairs for document
        positive_pairs += [(chunks[i], chunks[i+1]) 
                          for i in range(0, pairs_per_doc*2, 2)]
    
    return positive_pairs



def build_dataset(documents: list,
                  chunk_length: Optional[int] = 3,
                  max_sent: Optional[int] = None,
                  masking_percentage: Optional[float] = 0, 
                  max_pairs_per_doc: Optional[int] = None,
                  show_progress_bar: bool = True) -> list: 
    """
    Returns a list of masked positive pairs from a list of documents made of a 
    list of sentences. First, groups sentences into chunks of 'chunk_length' consecutive
    sentences. Then the pairs are built by randomly drawing pairs of chunks
    within each document without repetition. The output is flattened and each entry
    is a positive pair, so there is no separation between documents positive pairs. 
    Note, however, that it is not randomized, and thus all positive pairs of 
    document 1 will appear before all document 2 positive pairs, and so on. 
    
        Parameters: 
            documents (list of str): List of documents. Each document is itself
                a list of its sentences
            chunk_length (int) (def. 3): Length of each chunks.
            max_sent (int) (def. None): Maximum number of sentences
                to take into account for the chunks. Default None,
                which takes all sentences. 
            masking_percentage (float) (def. 0): float between 0 and 1 indicating
                the percentage of noun masking to apply. By default it does not
                apply any.
            max_pairs_per_doc (int) (def. None): maximum number of positive pairs
                to produce per document. By default builds document_length // 2 
                pairs. If the number given is bigger than the default value, it
                ignores it.
            show_progress_bar (bool) (def. True): Whether to show a progress bar.

        Returns: 
            positive_pairs (list): list of positive pairs. 
    """


    doc_chunks = [_build_chunks(doc, chunk_length, max_sent) for doc in documents]

    dataset = _build_dataset_from_chunks(doc_chunks,
                  masking_percentage,
                  max_pairs_per_doc,
                  show_progress_bar)
    return dataset
                  
