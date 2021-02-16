#*********************************************************************** PACKAGES ************************************************************************

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#********************************************************************** FUNCTIONS ************************************************************************



    #************************************ 1. COUNT STOP WORDS ************************************

#function to check the number of times each stop word is present in comments column

#if, for instance, I consider, due to the nature of the comments and the times a stop word appears
#in the comments, I'd remove that stop word from "stop_words" list
def count_stop_words(words=np.ndarray, comments=np.ndarray):
    stop_w_count = {} 

    for w in words:
        stop_w_count[w] = sum([c.split().count(w) for c in comments])

    return sorted(stop_w_count.items(), key=lambda t: t[1], reverse=True)


    #************************************ 2. LANGUAGE DETECTION ************************************

import langdetect
import cld3

#function to detect the language of the comments
#cld3 is a neural net trained by a google, langdetect is an individual package previously trained too
def dect_lang(comments=np.ndarray, detector="cld3"):
    language = []
    
    for comment in comments:
        if detector == "langdetect":
            try:
                language.append(langdetect.detect(comment))
            except Exception:
                language.append("NO LANG")
                
        elif detector == "cld3":
            try:
                language.append(cld3.get_language(comment)[0])
            except Exception:
                language.append("NO LANG")
                
    return language


    #************************************ 3. CLEAN SENTENCES ************************************

import re

#function to clean comments
def clean_comments(comments=np.ndarray, remove_accents="yes", bert_like="no"):
    _clean_comments = []
    
    for c in comments:
        clean_comment = re.sub(r"_|-", " ", c)
        clean_comment = re.sub(r"!", "", clean_comment)
        #remove digits
        clean_comment = re.sub(r"\d*", "", clean_comment)
        
        #remove dots & commas (for transformers like BERT, it is not recommended to remove dots neither commas)
        if bert_like == "no":
            clean_comment = re.sub(r"\.|,", "", clean_comment)
        #remove parentheses
        clean_comment = re.sub(r"\(|\)", "", clean_comment)
        clean_comment = clean_comment.lower()
        #remove spaces at the beginning and end
        clean_comment = clean_comment.strip()
        
        #for nlp, it is recommended to remove accents
        if remove_accents == "yes":
            clean_comment = re.sub(r"[àáâãäå]", 'a', clean_comment)
            clean_comment = re.sub(r"[èéêë]", 'e', clean_comment)
            clean_comment = re.sub(r"[ìíîï]", 'i', clean_comment)
            clean_comment = re.sub(r"[òóôõö]", 'o', clean_comment)
            clean_comment = re.sub(r"[ùúûü]", 'u', clean_comment)
        
        #we avoid to have empty rows
        if clean_comment != "":
            _clean_comments.append(clean_comment)
            
    return _clean_comments


    #************************************ 4. ADD NON-ACCENTS TO STOP WORDS ************************************

def add_non_accents(words):
    clean_stopwords = []

    for word in words:
        clean_words = re.sub(r"[àáâãäå]", 'a', word)
        clean_words = re.sub(r"[èéêë]", 'e', clean_words)
        clean_words = re.sub(r"[ìíîï]", 'i', clean_words)
        clean_words = re.sub(r"[òóôõö]", 'o', clean_words)
        clean_words = re.sub(r"[ùúûü]", 'u', clean_words)
        clean_stopwords.append(clean_words)

    return set(words + clean_stopwords)


    #************************************ 5. PREPROCESS ************************************

from gensim.models import Phrases
from gensim.models.phrases import Phraser

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.tokenize import word_tokenize

#function to preprocess comments in a tokenized list with bigrams & trigrams 
#(bigrams & trigrams will only appear in specific cases as; new york --> new_york)

#1. set root to "no" if you don't want the root (lemma) of the word, but the original world (default="yes")
#2. set stop_w to "yes" if you want to include stop_words (default="no")
def preprocess(comments=np.ndarray, root="yes", stop_w="no", stop_words=np.ndarray):
    
    tokens_list = []
    c_len = len(comments)
    
    for i in range(c_len):
        #create list of words from strings
        tokens = word_tokenize(comments[i])
        #remove token if not alpha
        tokens = [w for w in tokens if w.isalpha()]
        if root == "yes":
            #lemmatize finds the root of the word (is better than stemming in which the root could be a non real word)
            tokens = [lemmatizer.lemmatize(a) for a in tokens]
        if stop_w == "no":
            #remove stop words
            tokens = [w for w in tokens if w not in stop_words]
        #remove sentence with length 1 or 0 
        if len(tokens) > 1:
            tokens_list.append(tokens)
        
    bigram = Phrases(tokens_list, min_count=5, threshold=50) # higher threshold fewer phrases.
    trigram = Phrases(bigram[tokens_list], threshold=50)  
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    
    texts = [bigram_mod[doc] for doc in tokens_list]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    return texts


    #************************************ 6. VECTORIZER ************************************

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
      
#function to obtain term frequency / tf-idf

#points to take into account
#1. you can choose between Count or TF-idf vectorizers
#2. you can choose the ngrams

def basic_vectorizer(vectorizer=str, ngrams_range=(int, int), arr=np.ndarray, remove_accents="yes"):
    if vectorizer == "CountVectorizer":
        vectorizer = CountVectorizer
    elif vectorizer == "TfidfVectorizer":
        vectorizer = TfidfVectorizer
    else:
        vectorizer = None
    
    #the fn raises an error if the vectorizer is not count or tf-idf
    if vectorizer != CountVectorizer and vectorizer != TfidfVectorizer:
        raise ValueError("Please select between CountVectorizer / TfidfVectorizer")
    
    #setting params
    vectorize = vectorizer(ngram_range=ngrams_range) 
    #getting matrix of counts / tf-idf
    matrix = vectorize.fit_transform(arr)
    #getting terms (unique)
    features_names = [f.replace(" ", "_") for f in vectorize.get_feature_names()]

    #option to remove accents
    features = []
    
    if remove_accents == "yes":   
        for f in features_names: 
            clean_comment = re.sub(r"[àáâãäå]", 'a', f)
            clean_comment = re.sub(r"[èéêë]", 'e', clean_comment)
            clean_comment = re.sub(r"[ìíîï]", 'i', clean_comment)
            clean_comment = re.sub(r"[òóôõö]", 'o', clean_comment)
            clean_comment = re.sub(r"[ùúûü]", 'u', clean_comment)
            features.append(clean_comment)
    
    else:
        features = features_names
    
    #matrix to arr
    scores = matrix.toarray()

    #sum matrix counts / tf-idf
    matrix_sum = sum(np.array(scores))
    
    #getting original n terms, as features = set(terms)
    if vectorizer == CountVectorizer:
        terms = [((f+" ")*n).split() for f, n in zip(features, matrix_sum)]
        terms = [w for word in terms for w in word]
        
    #creating df with terms and their freqs
    data = {"term":[], "ranking":[]}
    tf_data = {}
    
    for index, term in enumerate(features): 
        data["term"].append(term) 
        data["ranking"].append(matrix_sum[index])
        
        #if tf-idf we need a dict of freqs for the wordcloud
        if vectorizer == TfidfVectorizer:
            tf_data[term] = matrix_sum[index]
            
    if vectorizer == CountVectorizer:
        #terms are the new tokens after count_vec mupliplied by their number of times
        return terms, pd.DataFrame(data)
    else:
        #tf_data is a dict of terms : freqs, needed for the wordcloud
        return tf_data, pd.DataFrame(data)


    #************************************ 7. WORDCLOUD ************************************

from wordcloud import WordCloud

#function to make a wordcloud

#points to take into account:
#1. set the title (str); if comments (input) are Negative Spanish comments, set it as tittle
#2. if your terms are from tf-idf vectorizer, set vectorizer = "tf-idf", else = None (default)
#3. if your terms are from tf-idf, fill freq_dict with terms (freq_dict=terms), else let freq_dict=None
#4. if your terms are not from tf-idf, fill comments = terms

def make_cloud(vectorizer=None, comments=None, freq_dict=None, title=None):
    
    if vectorizer == "tf-idf":
        wordcloud = WordCloud(width=800, height=560,
                              background_color='black', collocations=False,
                              min_font_size=10).generate_from_frequencies(freq_dict)
    else:
        wordcloud = WordCloud(width=800, height=560,
                              background_color='black', collocations=False,
                              min_font_size=10).generate(" ".join(comments))
    
    # plot the WordCloud image
    plt.figure(figsize=(15, 7), facecolor=None)
    plt.imshow(wordcloud)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=0)


    #************************************ 8. PLOT ************************************

def freq_graph(df=pd.DataFrame, length=int, figsize=(int, int)):
    plt.figure(figsize=figsize)
    
    plt.bar(df.sort_values(by="ranking", ascending=False)["term"][:length],
            df.sort_values(by="ranking", ascending=False)["ranking"][:length],
            align='center', alpha=0.5, width=0.75)
    
    plt.xticks(rotation="vertical")
    plt.ylabel('Frequency')
    plt.xlabel('Terms')
    plt.title('Frequency Terms Graph')
    plt.show()