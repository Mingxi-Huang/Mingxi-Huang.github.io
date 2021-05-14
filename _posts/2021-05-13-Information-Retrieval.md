---
layout: post
title: "TFIDF and Cosine Similarity"
subtitle: "search quotes from the transcript of \"Keeping up with the Kardashians\""
background: ''
---

# Assignment 4: "Search your transcripts. You will know it to be true." (Part 2)

## © Cristian Danescu-Niculescu-Mizil 2021

## CS/INFO 4300 Language and Information

**Learning Objectives**

- Develop an understanding of the inverted index and its applications
- Explore use cases of boolean search
- Examine how the inverted index can be used to efficiently compute IDF values
- Introduce cosine similarity as an efficient search model


```python
from collections import defaultdict
from collections import Counter
import json
import math
import string
import time
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from IPython.core.display import HTML
```


```python
with open("kardashian-transcripts.json", "r") as f:
    transcripts = json.load(f)
print(len(transcripts[0]))
```

    851



```python
treebank_tokenizer = TreebankWordTokenizer()
flat_msgs = [m for transcript in transcripts for m in transcript]
queries = [u"Honey, shea  the  one that walked in  with  a $5,000 all-black basic dress.",
           u"What is Kim doing?",
           u"Even at my advanced age, I got the hot wife, I got 2 great families.",
           u"I need a drink."]
```

## Finding the most similar messages (cosine similarity)

### A high-level overview

Our overall goal of the last part of this assignment is to build a system where we can compute the cosine similarity between queries and our datasets quickly. To accomplish queries and compute cosine similarities, we will need to represent documents as vectors. A common method of representing documents as vectors is by using "term frequency-inverse document frequency" (tf-idf) scores. More details about this method can be found [on the course website](https://canvas.cornell.edu/courses/27497/files?preview=3104608). The notation here is consistent with the hand out, so if you haven't read over it -- you should!

Consider the tf-idf representation of a document and a query: $\vec{d_j}$ and $\vec{q}$, respectively. Elements of these vectors are very often zero because the term frequency of most words in most documents is zero. Stated differently, most words don't appear in most documents! Consider a query that has 5 words in it and a vocabulary that has 20K words in it -- only .025% of the elements of the vector representation of the query are nonzero! When a vector (or a matrix) has very few nonzero entries, it is called "sparse." We can take advantage of the sparsity of tf-idf document representations to compute cosine similarity quickly. We will first build some data stuctures that allow for faster querying of statistics, and then we will build a function that quickly computes cosine similarity between queries and documents.

### A starting point
We will use an **inverted index** for efficiency. This is a sparse term-centered representation that allows us to quickly find all documents that contain a given term.

## Q1 Write a function to construct the inverted index (Code Completion)

As in class, the inverted index is a key-value structure where the keys are terms and the values are lists of *postings*. In this case, we record the documents a term occurs in as well as the **count** of that term in that document.


```python
def build_inverted_index(msgs):
    """ Builds an inverted index from the messages.
    
    Arguments
    =========
    
    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.
    
    Returns
    =======
    
    inverted_index: dict
        For each term, the index contains 
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]
        
    Example
    =======
    
    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])
    
    >> test_idx['be']
    [(0, 2), (1, 2)]
    
    >> test_idx['not']
    [(0, 1)]
    
    Note that doc_id refers to the index of the document/message in msgs.
    """
    # YOUR CODE HERE
    n = len(msgs)
    result = dict()
    for i in range(n):
        wc = dict()
        for word in msgs[i]['toks']:
            wc[word] = wc.setdefault(word, 0)+1
        for word, count in wc.items():
            result.setdefault(word, []).append((i, count))
    return result
                
```


```python
# This is an autograder test. Here we can test the function you just wrote above.
start_time = time.time()
inv_idx = build_inverted_index(flat_msgs)
execution_time = time.time() - start_time

assert len(inv_idx) <= 10000 
assert [i[0] for i in inv_idx['bruce']] == sorted([i[0] for i in inv_idx['bruce']])
assert len(inv_idx['bruce']) < len(inv_idx['kim'])
assert len(inv_idx['bruce']) >= 400 and len(inv_idx['bruce']) <= 435
assert len(inv_idx['baby']) >= 250 and len(inv_idx['baby']) <= 300
assert execution_time <= 1.0
```

## Q2 Using the inverted index for boolean search (Code Completion)

In this section we will use the inverted index you constructed to perform an efficient boolean search. The boolean model was one of the early information retrieval models, and continues to be used in applications today.

A boolean search works by searching for documents which match the boolean expression of the query. Three main operators in a boolean search are `AND` `OR` and `NOT`. For example, the query `"Ned" and "Rob"` would return any document which contains both the words "Ned" and "Rob".

Here, we will treat a query as a simple two-word search with exclusion. For example, the query words "kardashian", "kim" would be equivalent to the boolean expression `"kardashian" NOT "kim"`.

#### In class, we discussed the Merge Postings Algorithm. You can review the algorithm [here](https://canvas.cornell.edu/courses/27497/files?preview=3125065); code is provided [here](https://canvas.cornell.edu/courses/27497/files?preview=3125064) under "Posting merging algorithm for efficient boolean search".

The Merge Postings Algorithm we implemented can be thought of a boolean search with the `AND` operator. Write a function `boolean_search` that implements a similar algorithm with the `NOT` operator using the inverted index.

**Note:** Make sure you convert the `query_word` and `excluded_word` to lowercase. 

**Note:** We highly recommend that you implement the merge postings algorithm for `boolean_search`. You don't have to follow it exactly, but we may test whether your implementation is as efficient as the merge postings algorithm. We’d caution against trying a different implementation.

For your convenience, we provide below the algorithm you should implement.
------------------------------------------
    Initialize empty list (called merged list M)

    Create sorted list A of documents containing the query_word

    Create sorted list B of documents containing the excluded_word

    Start: Pointer at the first element of both A and B

    Do: Does it point to the same document ID in each list?

        Yes: advance pointer in both A and B
    
        No: 
            If the pointer with the smaller document ID is in list A:
                Append the smaller document ID to list M
            
            Advance (to the right) the pointer with the smaller ID
    
    End: When we attempt to advance a pointer already at the end of its list

    Finally: if there are remaining document IDs in list A that were not evaluated in the above loop, then append them to list M.

------------------------------------------

**Note:** The objective is to demonstrate your knowledge in building an efficient search algorithm. If you use the Python `set.difference` function, you will lose points.



```python
def boolean_search(query_word,excluded_word, inverted_index):
    """ Search the collection of documents for the given query_word 
        provided that the documents do not include the excluded_word
    
    Arguments
    =========
    
    query_word: string,
        The word we are searching for in our documents.
    
    excluded_word: string,
        The word excluded from our documents.
    
    inverted_index: an inverted index as above
    
    
    Returns
    =======
    
    results: list of ints
        Sorted List of results (in increasing order) such that every element is a `doc_id`
        that points to a document that satisfies the boolean
        expression of the query.
        
    """
    # YOUR CODE HERE
    query_word = query_word.lower()
    excluded_word = excluded_word.lower()
    M = []
    A, B = [], []
    for doc_id, count in inverted_index[query_word]:
        if count > 0:
            A.append(doc_id)
    for doc_id, count in inverted_index[excluded_word]:
        if count > 0:
            B.append(doc_id)
    i, j = 0, 0
    while i < len(A) and j < len(B):
        if A[i] == B[j]: 
            i += 1
            j += 1
        else:
            if A[i] < B[j]:
                M.append(A[i])
                i += 1
            else:
                j += 1
    if i <= len(A) - 1:
        for k in range(i, len(A)):
            M.append(A[i])
    return M
```


```python
result0_start_time = time.time()
result0 = boolean_search('ice','cream', inv_idx)
result0_execution_time = time.time() - result0_start_time
result3 = boolean_search('puppy','dog', inv_idx)
result1= boolean_search('Kardashian','Kim',inv_idx)
result4= boolean_search('cake','cake',inv_idx)
result5= boolean_search('honey','money',inv_idx)
assert result0_execution_time < 1.0
assert type(result1) == list
assert len(result3) == 7
assert len(result4)==0
assert len(result5)==237
```

## Q2b Using the inverted index for boolean search (Free Response)

In A3 we already explored search techniques which are able to find a wider variety of relevant results. Why might you want to use a boolean search with an inverted index instead? Give a specific example in which a boolean search would be a better choice than a search with edit distance, and justify why a boolean search would be preferable.

<div style="border-bottom: 4px solid #AAA; padding-bottom: 6px; font-size: 16px; font-weight: bold;">Write your answer in the provided cell below</div>

For example, if we want Kardashian but not Kim, we may want some Kardashian family members other than Kim Kardashian. With edit distance, we don't have a very good way other than search all members one by one (compute each edit distance of Khloe Kardashian, Kourtney Kardashian, etc.) But with boolean search this is very easy, by definition. 

<div style="border-top: 4px solid #AAA; padding-bottom: 6px; font-size: 16px; font-weight: bold; text-align: center;"></div>

## Q3 Compute IDF *using* the inverted index (Code Completion)

Write a function `compute_idf` that uses the inverted index to efficiently compute IDF values.

Words that occur in a very small number of documents are not useful in many cases, so we ignore them. Use a parameter `min_df`
to ignore all terms that occur in strictly fewer than `min_df=15` documents.

Similarly, words that occur in a large *fraction* of the documents don't bring any more information for some tasks. Use a parameter `max_df_ratio` to trim out such words. For example, `max_df_ratio=0.90` means ignore all words that occur in more than 90% of the documents.

As a reminder, we define the IDF statistic as...
$$ IDF(t) = \log \left(\frac{N}{1 + DF(t)} \right) $$

where $N$ is the total number of docs and $DF(t)$ is the number of docs containing $t$. Keep in mind, there are other definitions of IDF out there, so if you go looking for resources on the internet, you might find differing (but also valid) accounts. In practice the base of the log doesn't really matter, however you should use base 2 here.

**Note:** If words are ignored due to appearing too frequently or not frequently enough, they should not be added to the `idf` dicitionary.


```python
def compute_idf(inv_idx, n_docs, min_df=15, max_df_ratio=0.90):
    """ Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.
    
    Hint: Make sure to use log base 2.
    
    Arguments
    =========
    
    inv_idx: an inverted index as above
    
    n_docs: int,
        The number of documents.
        
    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored. 
        Documents that appear min_df number of times should be included.
    
    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.
    
    Returns
    =======
    
    idf: dict
        For each term, the dict contains the idf value.
        
    """
    
    # YOUR CODE HERE
    idf = dict()
    for word, tuples in inv_idx.items():
        df_word = len(tuples)
        df_ratio = df_word / n_docs
        if min_df > df_word or  max_df_ratio < df_ratio:
            continue
        idf[word] = np.log2(n_docs / (1 + df_word))
    return idf
        
        
```


```python
# This is an autograder test. Here we can test the function you just wrote above.
start_time = time.time()
idf_dict = compute_idf(inv_idx, len(flat_msgs))
execution_time = time.time() - start_time

assert len(idf_dict) < len(inv_idx)
assert 'blah' not in idf_dict
assert 'blah' in inv_idx 
assert '.' in idf_dict
assert '3' not in idf_dict
assert idf_dict['bruce'] >= 6.0 and idf_dict['bruce'] <= 7.0
assert idf_dict['baby'] >= 6.0 and idf_dict['baby'] <= 8.0
assert execution_time <= 1.0
```

## Q4 Compute the norm of each document using the inverted index (Code Completion)

Recalling our tf-idf vector representation of documents, we can compute the "norm" as the norm (length) of the vector representation of that document. More specifically, the norm of a document $j$, denoted as $\left|\left| d_j \right|\right|$, is given as follows...

$$ \left|\left| d_j \right|\right| = \sqrt{\sum_{\text{word } i} (tf_{ij} \cdot idf_i)^2} $$

This will be important for the following question, where it is one of the required components for computing cosine similarity between a query and a document. 

**Note:** Please use the above formula to compute the norm, and not any other formulae e.g. those from in-class quizzes.


```python
def compute_doc_norms(index, idf, n_docs):
    """ Precompute the euclidean norm of each document.
    
    Arguments
    =========
    
    index: the inverted index as above
    
    idf: dict,
        Precomputed idf values for the terms.
    
    n_docs: int,
        The total number of documents.
    
    Returns
    =======
    
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    
    # YOUR CODE HERE
    norms = np.zeros(n_docs)
    for word, idf_score in idf.items():
        for doc_id, count in index[word]:
            norms[doc_id] += (count * idf_score) ** 2
    return np.sqrt(norms)
```


```python
# This is an autograder test. Here we can test the function you just wrote above.
start_time = time.time()
doc_norms = compute_doc_norms(inv_idx, idf_dict, len(flat_msgs))
execution_time = time.time() - start_time

assert len(flat_msgs) == len(doc_norms)
assert doc_norms[3722] == 0
assert max(doc_norms) < 80
assert doc_norms[1] >= 15.5 and doc_norms[1] <= 17.5
assert doc_norms[5] >= 6.5 and doc_norms[5] <= 8.5
assert execution_time <= 1.0
```

## Q5 Find the most similar messages to the quotes (Code Completion)

The goal of this section is to implement `index_search`, a fast implementation of cosine similarity. You will then test your answer by running the search function using the code provided. Briefly discuss why it worked, or why it might not have worked, for each query.

The goal of `index_search` is to compute the cosine similarity between the query and each document in the dataset. Naively, this computation requires you to compute dot products between the query tf-idf vector $q$ and each document's tf-idf vector $d_i$.

However, you should be able to use the sparsity of the tf-idf representation and the data structures you created to your advantage. More specifically, consider the cosine similarity...

$$ cossim(\vec{q}, \vec{d_j}) = \frac{\vec{q} \cdot \vec{d_j}}{\|\vec{q}\| \cdot \|\vec{d_j}\|}$$

Specifically, focusing on the numerator...

$$ \vec{q} \cdot \vec{d_j} = \sum_{i} {q_i} * {d_i}_j $$

Here ${q_i}$ and ${d_i}_j$ are the $i$-th dimension of the vectors $q$ and ${d_j}$ respectively.
Because many ${q_i}$ and ${d_i}_j$ are zero, it is actually a bit wasteful to actually create the vectors $q$ and $d_j$ as numpy arrays; this is the method that you saw in class.

A faster approach to computing the numerator term of cosine similarity involves quickly computing the above summation using the inverted index, pre-computed idf scores, and pre-computed document norms.

A good "first step" to implementing this efficiently is to only loop over ${q}_j$ that are nonzero (i.e. ${q}_j$ such that the word $j$ appears in the query). 

**Note:** Convert the query to lowercase, and use the `nltk.tokenize.TreebankWordTokenizer` to tokenize the query (provided to you as the `tokenizer` parameter). The transcripts have already been tokenized this way. <br>

**Note:** For `index_search`, you need not remove punctuation tokens from the tokenized query before searching.

**Note:** It is okay to see some duplicates in your printed results.

**Aside:** Precomputation

In many settings, we will need to repeat the same kind of operation many times. Often, part of the input doesn't change.
Queries against the Kardashians transcript are like this: we want to run more queries (in the real world we'd want to run a lot of them every second, even) but the data we are searching doesn't change.

We could write an `index_search` function with the same signature as A3's `verbatim_search`, taking the `query` and the `msgs` as input, and the function would look like:

    def index_search(query, msgs):
        inv_idx = build_inverted_index(msgs)
        idf = compute_idf(inv_idx, len(msgs))
        doc_norms = compute_doc_norms(inv_idx)
        # do actual search


But notice that the first three lines only depend on the messages. Imagine if we run this a million times with different queries but the same collection of documents: we'd wastefully recompute the index, the IDFs and the norms every time and discard them. It's a better idea, then, to precompute them just once, and pass them as arguments.


```python
inv_idx = build_inverted_index(flat_msgs)

idf = compute_idf(inv_idx, len(flat_msgs),
                  min_df=15,
                  max_df_ratio=0.1)  # documents are very short so we can use a small value here
                                     # examine the actual DF values of common words like "the"
                                     # to set these values

inv_idx = {key: val for key, val in inv_idx.items()
           if key in idf}            # prune the terms left out by idf

doc_norms = compute_doc_norms(inv_idx, idf, len(flat_msgs))
```


```python
def index_search(query, index, idf, doc_norms, tokenizer=treebank_tokenizer):
    """ Search the collection of documents for the given query
    
    Arguments
    =========
    
    query: string,
        The query we are looking for.
    
    index: an inverted index as above
    
    idf: idf values precomputed as above
    
    doc_norms: document norms as computed above
    
    tokenizer: a TreebankWordTokenizer
    
    Returns
    =======
    
    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score (descending order), but if there is 
        a tie for the score, sort by the second element, that is
        the `doc_id` with ascending order. 
        An example is as follows:
        
        score       doc_id
       [(0.9,       1000),
        (0.9,       1001),
        (0.8,       2000),
        (0.8,       2001),
        (0.8,       2002),
        ...]

        
    """
    
    # YOUR CODE HERE
    tok_q = tokenizer.tokenize(query.lower())
    
    query_tf = dict()
    for word in tok_q:
        query_tf[word] = query_tf.setdefault(word, 0) + 1
    
    index_dict = dict()
    for word in index:
        index_dict[word] = dict()
        for doc_id, count in index[word]:
            index_dict[word][doc_id] = count
        
    result = []        
    for doc_id in range(len(doc_norms)):
        qnorm = 0
        dotprod = 0
        for word in query_tf.keys():
            if word not in idf:
                continue
            query_tfidf = query_tf[word] * idf[word]
            doc_tfidf = index_dict[word].get(doc_id, 0) * idf[word]
            qnorm += query_tfidf ** 2
            dotprod += query_tfidf * doc_tfidf
        qnorm = np.sqrt(qnorm)
        denom = doc_norms[doc_id] * qnorm
        score = (dotprod / denom) if denom != 0 else 0
        result.append((score, doc_id))
    
    result = sorted(result, key = lambda x :(x[0], x[1]), reverse = True)
    print(result[:5])
    return result
    
```


```python
# This is an autograder test. Here we can test the function you just wrote above.
"""
Note that in the printing part, It is ok for there to be duplicates.
"""
start_time = time.time()
results = index_search(queries[1], inv_idx, idf, doc_norms)
execution_time = time.time() - start_time

assert type(results[0]) == tuple
assert max(results)[0] == results[0][0]
assert results[0][0] >= 0.65 and results[0][0] <= 0.80
assert execution_time <= 1.0

for i in range(1,len(queries)):
    print("#" * len(queries[i]))
    print(queries[i])
    print("#" * len(queries[i]))

    for score, msg_id in index_search(queries[i], inv_idx, idf, doc_norms)[:10]:
        print("[{:.2f}] {}: {}\n\t({})".format(
            score,
            flat_msgs[msg_id]['speaker'],
            flat_msgs[msg_id]['text'],
            flat_msgs[msg_id]['episode_title'])) 
    print()
```

    [(0.7387080125279152, 33110), (0.7387080125279152, 24653), (0.7387080125279152, 13196), (0.7387080125279152, 9792), (0.7387080125279152, 4971)]
    ##################
    What is Kim doing?
    ##################
    [(0.7387080125279152, 33110), (0.7387080125279152, 24653), (0.7387080125279152, 13196), (0.7387080125279152, 9792), (0.7387080125279152, 4971)]
    [0.74] BRUCE: What you doing?
    	(The Wedding: Keeping Up With the Kardashians)
    [0.74] LAMAR: What you doing, mama?
    	(Keeping Up With the Kardashians - Blind Date)
    [0.74] KOURTNEY: Doing what?
    	(Keeping Up With the Kardashians - Kris the Cheerleader)
    [0.74] KHLOE: Doing what?
    	(Keeping Up With the Kardashians - Birthday Suit)
    [0.74] BRUCE: Doing what?
    	(Keeping Up With the Kardashians - Baby Blues)
    [0.68] KHLOE: What is this doing for you?
    	(Keeping Up With the Kardashians - Shape Up or Ship Out)
    [0.67] KOURTNEY: Kim is a hopeless romantic.
    	(Keeping Up With the Kardashians - Kris the Cheerleader)
    [0.63] KOURTNEY: What are you doing?
    	(Kourtney and Kim Take New York - Life in the Big City)
    [0.63] KOURTNEY: What are you doing?
    	(Kourtney and Kim Take New York - Diva Las Vegas)
    [0.63] SCOTT: What are you doing?
    	(Kourtney and Kim Take New York - Down and Out in New York City)
    
    ####################################################################
    Even at my advanced age, I got the hot wife, I got 2 great families.
    ####################################################################
    [(0.9568071434973142, 4259), (0.4876972035565681, 28559), (0.4710509075425327, 28993), (0.46134674413388577, 36308), (0.46134674413388577, 35654)]
    [0.96] BRUCE: Even at my age, I got the hot wife, I got a great family.
    	(Keeping Up With the Kardashians - Kardashian Family Vacation)
    [0.49] KIM: I got it at Madison.
    	(Keeping Up With the Kardashians - Double Trouble)
    [0.47] KRIS: You got my vote.
    	(Keeping Up With the Kardashians - Double Trouble)
    [0.46] BRUCE: You got it.
    	(Keeping Up With the Kardashians - I'm Watching You)
    [0.46] KRIS: You got it?
    	(Keeping Up With the Kardashians - Helping Hand)
    [0.46] ROB: I got it, I got it.
    	(Keeping Up With the Kardashians - Helping Hand)
    [0.46] WOMAN: Got it.
    	(The Wedding: Keeping Up With the Kardashians)
    [0.46] SHARON: Got it?
    	(The Wedding: Keeping Up With the Kardashians)
    [0.46] KRIS: I got it, I got it.
    	(Keeping Up With the Kardashians - Rob's New Girlfriend)
    [0.46] MALIKA: Got it, got it.
    	(Keeping Up With the Kardashians - Kourt Goes A.W.O.L.)
    
    ###############
    I need a drink.
    ###############
    [(1.0000000000000002, 9949), (1.0000000000000002, 4268), (0.7767752521378065, 14401), (0.7534932063864643, 9411), (0.7478153117950368, 1139)]
    [1.00] KHLOE: I need a drink.
    	(Keeping Up With the Kardashians - Birthday Suit)
    [1.00] KHLOE: I need a drink.
    	(Keeping Up With the Kardashians - Kardashian Family Vacation)
    [0.78] KRIS: Have a drink.
    	(Keeping Up With the Kardashians - Pussycat Vision)
    [0.75] KRIS: I want to drink a toast to Nilda and Joe.
    	(Keeping Up With the Kardashians - Leaving the Nest)
    [0.75] KRIS: No wonder I drink.
    	(Keeping Up With the Kardashians - Shape Up or Ship Out)
    [0.75] KRIS: No wonder I drink.
    	(Keeping Up With the Kardashians - Shape Up or Ship Out)
    [0.70] ADRIENNE: Get me a drink.
    	(The Wedding: Keeping Up With the Kardashians)
    [0.70] ADRIENNE: Get me a drink.
    	(Keeping Up With the Kardashians - The Wedding)
    [0.65] SCOTT: I don't really have a drink.
    	(Kourtney and Kim Take New York - Down and Out in New York City)
    [0.65] SCOTT: I don't really have a drink.
    	(Kourtney and Kim Take New York - Start Spreading the News)
    


## Q5b Find the most similar messages to the quotes (Free Response)

Briefly discuss why cosine similarity worked, or why it might not have worked, **for each query**.

<div style="border-bottom: 4px solid #AAA; padding-bottom: 6px; font-size: 16px; font-weight: bold;">Write your answer in the provided cell below</div>

1. It's not working. From the fact that the first five results have the same score, we can infer that "Kim" and "is" may be removed by our cutoff scheme, so only matching two words cannot find exactly what we want. Also, the word order doesn't matter in calculating our score also leads to the results we get.
2. It's working. From the fact that the difference between the firs two results is high, we see that our algorithm distinguished what is close and what is not. There are less common words in the query, and the query is long enough for data to be more accurate. From this we can genralize that the length of the query and the uniqueness of the word affect our score a lot.
3. It's working. We are able to fetch the exact same sentence, and as we can see the score is 1.00, cos(0) = 1, there's not difference between what is found and the query. For the less relevant documents, even though the score is not perfect, the score is high. This is because "drink" is not very common and the documents that contain the word are all somewhat relevant.

<div style="border-top: 4px solid #AAA; padding-bottom: 6px; font-size: 16px; font-weight: bold; text-align: center;"></div>

## Q6EC: Extra credit question 1 (optional)

### Updating precomputed values.

In many real-world applications, the collection of documents will not stay the same forever. At Internet-scale, however, it could possibly even be worth recomputing things every second, if during that second we're going to answer millions of queries.

However, there's a better way: in reality, the document set will not change radically, but incrementally.  In particular, it's most common to add or remove a bunch of new documents to the index.

Write functions `add_docs` and `remove_docs` that update the index, idf and document norms.  Think of the implications this has on how we store the IDF. Is there a better way of storing it, that minimizes the memory we need to touch when updating?

Think of adequate test cases for these functions and implement them.

**Note:** You can get up to 0.5 EC for completing this question. *Do not delete the cell below.* Please comment out the `raise NotImplementedError()` if you choose not to answer this question. 


```python
# YOUR CODE HERE
    
# raise NotImplementedError()
```

## Q7EC: Extra credit question 2 (optional)

### Finding your own similarity metric

We've explored using cosine similarity and edit distance to find similar messages to input queries. However, there's a whole world of ways to measure the similarity between two documents. Go forth, and research!

(Fun fact: Fundamental information retrieval techniques were in fact developed at Cornell, so you would not be the first Cornellian to disrupt the field)

For this question, find a new way of measuring similarity between two documents, and implement a search using your new metric. Your new way of measuring document similarity should be different enough from the two approaches we already implemented. It can be a method you devise or an existing method from somewhere else (make sure to reveal your sources).

**Note:** The amount of EC awarded for this question will be determined based on creativity, originality, implementation, and analysis. *Do not delete the cell below.* Please comment out the `raise NotImplementedError()` if you choose not to answer this question. 


```python
# YOUR CODE HERE
# raise NotImplementedError()
```


```python

```
