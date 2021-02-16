#*********************************************************************** PACKAGES ************************************************************************

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


#********************************************************************** FUNCTIONS ************************************************************************



   #************************************ 1. CLEAN DATAFRAME ************************************

import re

#function to clean comments
def clean_df(df=pd.DataFrame, remove_accents="yes", bert_like="no"):

    df_clean = {"id":[], "area":[], "comment":[], "clean_comment":[], "comment_type":[], "language":[]}
    
    for i, a, c, i_c, l in zip(df.ID, df.Area, df.Comment, df.Initial_Classification, df.language):
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
            df_clean["id"].append(i)
            df_clean["area"].append(a)
            df_clean["comment"].append(c)
            df_clean["clean_comment"].append(clean_comment)
            df_clean["comment_type"].append(i_c)
            df_clean["language"].append(l)
                
    return pd.DataFrame(df_clean)


    #************************************ 2. PREPROCESS DATAFRAME ************************************

from gensim.models import Phrases
from gensim.models.phrases import Phraser

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.tokenize import word_tokenize

#function to preprocess comments in a tokenized list with bigrams & trigrams 
#(bigrams & trigrams will only appear in specific cases as; new york --> new_york)

#1. set root to "no" if you don't want the root (lemma) of the word, but the original world (default="yes")
#2. set stop_w to "yes" if you want to include stop_words (default="no")
def preprocess_df(df=pd.DataFrame, root="yes", stop_w="no", stop_words=np.ndarray):
    
    df_processed = {"id":[], "area":[], "comment":[], "clean_comment":[], "comment_type":[], "language":[]}
    
    tokens_list = []
    
    comments_range = range(len(df.clean_comment))
    comments = list(df.clean_comment)
    
    
    for c_r, i, a, c, ct, l in zip(comments_range, df.id, df.area, df.comment, df.comment_type, df.language):
        #create list of words from strings
        tokens = word_tokenize(comments[c_r])
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
            df_processed["id"].append(i)
            df_processed["area"].append(a)
            df_processed["comment"].append(c)
            df_processed["comment_type"].append(ct)
            df_processed["language"].append(l)
    
    #create bigrams and trigrams (the bigrams and trigrams created are well know ones like 'new_york')
    bigram = Phrases(tokens_list, min_count=5, threshold=50) # higher threshold fewer phrases.
    trigram = Phrases(bigram[tokens_list], threshold=50)  
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    
    texts = [bigram_mod[doc] for doc in tokens_list]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    #flatten texts
    df_processed["clean_comment"] = [" ".join(comments) for comments in texts]    
    
    return pd.DataFrame(df_processed)


    #************************************ 3. TFIDF DATAFRAME | COUNTVECTORIZE DATAFRAME ************************************

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
      
#function to obtain term frequency / tf-idf

#points to take into account
#1. you can choose between Count or TF-idf vectorizers
#2. you can choose the ngrams

def vectorizer_df(vectorizer=str, ngrams_range=(int, int), df=pd.DataFrame, remove_accents="yes", stop_w="no", stop_words=np.ndarray):
    if vectorizer == "CountVectorizer":
        vectorizer = CountVectorizer
    elif vectorizer == "TfidfVectorizer":
        vectorizer = TfidfVectorizer
    else:
        vectorizer = None
    
    #the fn raises an error if the vectorizer is not count or tf-idf
    if vectorizer != CountVectorizer and vectorizer != TfidfVectorizer:
        raise ValueError("Please select between CountVectorizer / TfidfVectorizer")
            
    arr = list(df.comment)
    
    #setting params
    if stop_w == "no":
        vectorize = vectorizer(ngram_range=ngrams_range, stop_words=stop_words) 
        
    else:
        vectorize = vectorizer(ngram_range=ngrams_range, stop_words=None)
        
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
            #remove digits
            clean_comment = re.sub(r"\d*", "", clean_comment)
            features.append(clean_comment)
    
    else:
        for f in features_names:
            clean_comment = re.sub(r"\d*", "", f)
            features.append(clean_comment)

    
            
    #matrix to arr
    scores = matrix.toarray()
    
    #this point is to create a df of the new grams and scores, keeping the original IDs 
    
    scores_dict = {}
    
    for i, ct, s in zip(list(df.id), list(df.comment_type), scores):
        scores_dict[(i, ct)] = s
        
    id_dict = {"id":[], "score":[], "feature":[], "comment_type":[]}
    
    for i, ids, values in zip(enumerate(scores_dict.values()), scores_dict.keys(), scores_dict.values()):
        for score, feature in zip(list(scores_dict.values())[i[0]], features):
            id_dict["id"].append(ids[0])
            id_dict["score"].append(score)
            id_dict["feature"].append(feature)
            id_dict["comment_type"].append(ids[1])
    
    df_id = pd.DataFrame(id_dict)
            
    keep_id = []
    
    for s, i in zip(df_id.score, df_id.id):
        if s == 0.0:
            keep_id.append(np.nan)
        else:
            keep_id.append(i)
            
    df_id.id = keep_id
            
    return df_id.groupby(["feature", "id", "comment_type"]).agg({"score":"sum"}).reset_index()
    


    #************************************ 4. WORDCLOUD ************************************

from wordcloud import WordCloud

#function to plot a wordcloud on a clustered df

def get_wordcloud(topic=None, model=None, df=pd.DataFrame, clusters=None, grams=str):

    if grams == "unigrams":
        tokens = " ".join([t.replace("_", " ") for t in list(df.comments[df.clusters==clusters])])
    
    elif grams == "computed":
        tokens = " ".join([t.replace(" ", "_") for t in list(df.comments[df.clusters==clusters])])
    
    try:
        wordcloud = WordCloud(width=800, height=560,
                            background_color='black', collocations=False,
                            min_font_size=10).generate(tokens)

        # plot the WordCloud image
        plt.figure(figsize=(15, 7), facecolor=None)
        plt.imshow(wordcloud)
        plt.title(f"Cluster Number {clusters}")
        plt.axis("off")
        plt.tight_layout(pad=0)
    
    except Exception:
        print(f"The wordcloud of the Cluster Number {clusters} cannot be generated due to it has too long sentences")


    #************************************ 5. SHILOUETTE SCORE SKLEARN (EUCLIDEAN) ************************************

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import normalize

#this function use kmeans with euclidean distance

def shilouette(vecs=np.ndarray, kmax=40, scale=str): 
    if scale == "yes":
        vecs = normalize(vecs, norm='l2')
    sil = []
    klusters = range(2, kmax+1)
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in klusters:
        km = KMeans(n_clusters=k).fit(vecs)
        labels = km.labels_
        sil.append(silhouette_score(vecs, labels, metric='euclidean'))

    plt.plot(klusters, sil, 'bx-')
    plt.xlabel('k')
    plt.ylabel('silhouette_score')
    plt.title('Silhouette Score Method For Optimal k')
    plt.show()


    #************************************ 6. SHILOUETTE SCORE TENSORFLOW (COSINE) ************************************

import tensorflow as tf
from tensorflow.python.ops import clustering_ops

#this function uses kmeans with cosine distance

SQUARED_EUCLIDEAN_DISTANCE = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE
COSINE_DISTANCE = clustering_ops.COSINE_DISTANCE

def shilouette_tf(input_fn, vecs=np.ndarray, kmax=40, distance_m=COSINE_DISTANCE, sil_metric="cosine"): 
    sil = []
    klusters = range(2, kmax+1)
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in klusters:
        km = tf.compat.v1.estimator.experimental.KMeans(num_clusters=k, distance_metric=distance_m, use_mini_batch=False)
        km.train(input_fn)
        labels = [p["cluster_index"] for p in list(km.predict(input_fn))]
        sil.append(silhouette_score(vecs, labels, metric=sil_metric))

    plt.plot(klusters, sil, 'bx-')
    plt.xlabel('k')
    plt.ylabel('silhouette_score')
    plt.title('Silhouette Score Method For Optimal k')
    plt.show()


    #************************************ 7. KMEANS SKLEARN (EUCLIDEAN) ************************************

#this function use kmeans with euclidean distance

def clusterK(vecs=np.ndarray, k=int, scale=str):
    if scale == "yes":
        vecs = normalize(vecs, norm='l2')
        
    km = KMeans(n_clusters=k).fit(vecs)
    clusters = km.predict(vecs)
        
    return clusters


    #************************************ 8. KMEANS TENSORFLOW (COSINE) ************************************

#this function use kmeans with cosine distance

def clusterK_tf(input_fn, k=int, distance_m=COSINE_DISTANCE):
    km = tf.compat.v1.estimator.experimental.KMeans(num_clusters=k, distance_metric=distance_m, use_mini_batch=False)
    km.train(input_fn)
    clusters = [p["cluster_index"] for p in list(km.predict(input_fn))]
    return clusters


    #************************************ 9. HDBSCAN CLUSTERING (EUCLIDEAN | COSINE) ************************************

from hdbscan import HDBSCAN

def clusterH(vecs=np.ndarray, min_cluster_size=5, min_samples=None, distance_m="euclidean"):
    if distance_m == "euclidean":
        hd = HDBSCAN(min_cluster_size=5, min_samples=None, metric="euclidean")
        clusters = hd.fit(vecs)
    
    elif distance_m == "cosine": 
        norm_data = normalize(vecs, norm='l2')
        hd = HDBSCAN(min_cluster_size=5, min_samples=None, metric="euclidean")
        clusters = hd.fit(norm_data)
        
    return clusters.labels_


    #************************************ 10. AGGLOMERATIVE CLUSTERING (EUCLIDEAN | COSINE) ************************************

from sklearn.cluster import AgglomerativeClustering

def cluster_agg(vecs=np.ndarray, k=int, metric=str):
    if metric == "cosine":
        agg = AgglomerativeClustering(n_clusters=5, affinity='cosine', linkage='single')
        
    elif metric == "euclidean":
        agg = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
        
    clusters = agg.fit_predict(vecs)
    
    return clusters


    #************************************ 11. STANDARD SCALER ************************************

from sklearn.preprocessing import StandardScaler

def scaler(x):
    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)
    
    return x_scaled


    #************************************ 12. PRINCIPAL COMPONENT ANALYSIS (PCA) ************************************

from sklearn.decomposition import PCA

def reducPca(x, n_components=3):
    pca = PCA()
    pca.fit(x)
    print("Top 10 Explained Variance Variables by percentage\n", pca.explained_variance_ratio_[:10],"\n")

    #fit and transform
    pca = PCA(n_components=3).fit(x)
    print("Explained Variance by Percentage with the 3 components selected", pca.explained_variance_ratio_,"\n\n")
    pca_3d = pca.transform(x)
    return pca_3d


    #************************************ 13. DATAFRAME CONCAT ************************************

#function to concatenate 3 DFs including the last with 3 dimensions

def dfConcat(data1=pd.DataFrame, name1=str, data2=pd.DataFrame, name2=str, data3=pd.DataFrame, name3=[]):
    df1 = pd.DataFrame(data1, columns=[name1])
    df2 = pd.DataFrame(data2, columns=[name2])
    df3 = pd.DataFrame(data3, columns=[name3[0], name3[1], name3[2]])    
    df = pd.concat([df1, df2, df3], axis=1)
    print(df.shape)
    print(df.head())
    return df


    #************************************ 14. VECTORS VISUALIZATION ************************************

#function to plot a 3d vectors in an interactive way from a df containing the clusters

import plotly.graph_objects as go
import plotly.offline as py

def visualize(df):
    cont = []
    for i in set(df["clusters"]):
        fig = go.Scatter3d(x=df[df["clusters"]==i]['x'],
                            y=df[df["clusters"]==i]['y'],
                            z=df[df["clusters"]==i]['z'],
                            text=df[df["clusters"]==i]['comments'],
                            marker=dict(opacity=0.9,
                                        reversescale=True,
                                        colorscale='Blues',
                                        size=5),
                            line=dict (width=0.02),
                            mode='markers')

        cont.append(fig)    

    #Make Plot.ly Layout
    mylayout = go.Layout(scene=dict(xaxis=dict( title="PCA1"),
                                    yaxis=dict( title="PCA2"),
                                    zaxis=dict(title="PCA3")),)

    #Plot and save html

    py.iplot({"data": cont,
                         "layout": mylayout},
                         filename=("3DPlot.html"))


    #************************************ 15. VECTORIZATION SENTENCE TRANSFORMERS ************************************

#function to load and vectorize pre-trained NLP models able to use directly from sentence_tranformers

from sentence_transformers import SentenceTransformer

def st_vectorize(comments=np.ndarray, model=str):
    bert = SentenceTransformer(model)
    vec = np.array(bert.encode(comments, show_progress_bar=True))
    return vec


    #************************************ 16. VECTORIZATION HUGGING FACE ************************************

#function to load and vectorize pre-trained NLP models NOT able to use directly from sentence_tranformers

from sentence_transformers import models

def hf_vectorize(comments=np.ndarray, model=str):
    
    if model == "bert":
        model = "bert-base-multilingual-cased"
        word_embedding_model = models.BERT(model)
    
    elif model == "distilbert":
        model = "distilbert-base-multilingual-cased"
        word_embedding_model = models.DistilBERT(model)
    
    else:
        raise ValueError("Please choose 'distilbert' or 'bert'")

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    mult_bert = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    mult_vecs = np.array(mult_bert.encode(comments, show_progress_bar=True))
    return mult_vecs


    #************************************ 17. SHOW CLUSTERS FREQUENCY ************************************

def show_cluster_freq(df=pd.DataFrame, image=str):
    plt.figure(figsize=(15,7))
    
    if image == "pie":
        plt.pie(df.clusters.value_counts().values, autopct='%1.1f%%')
        plt.legend(labels=df.clusters.value_counts().index, loc="best", bbox_to_anchor=(1, 1, 0, 0))
        plt.title("Cluster Distribution")
        
    elif image == "bar":
        plt.bar(df.clusters.value_counts().index,
                df.clusters.value_counts().values,
                align='center', alpha=0.5, width=0.75)

        plt.xticks(rotation="vertical")
        plt.ylabel('Frequency')
        plt.xlabel('Clusters')
        plt.title('Frequency Terms Graph')
        
    else:
        raise ValueError("This function just accepts 'pie' or 'bar' as images")
    
    plt.show()


    #************************************ 18. SHOW DENDROGRAM ************************************

from scipy.cluster.hierarchy import dendrogram, linkage

def get_dendrogram(vecs=np.ndarray, metric=str, p=int, labels=np.ndarray, i=int):
    
    linked = linkage(vecs, method="single", metric=metric)

    plt.figure(figsize=(10, 7))

    dendrogram(linked,
            p=p,
            truncate_mode="level",
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True)
    
    plt.title(f"Dendogram for cluster {i}")
    plt.xticks(rotation="vertical", fontsize="medium")
    plt.show()


    #************************************ 19. TOKENS SELECTION ************************************

def tokens_selection(df=pd.DataFrame, element=str):
    if element == "df_ngrams":
        return list(set(df.feature))
    
    elif element == "positive_ngrams":
        return list(set(df.feature[df.comment_type=="Positivo"]))
    
    elif element == "negative_ngrams":
        return list(set(df.feature[df.comment_type=="Negativo"]))
    
    elif element == "df_clean_comments":
        return list(set(df.clean_comment))
    
    elif element == "positive_clean_comments":
        return list(set(df.clean_comment[df.comment_type=="Positivo"]))
    
    elif element == "negative_clean_comments":
        return list(set(df.clean_comment[df.comment_type=="Negativo"]))
    
    else:
        raise ValueError("Please select 'df_ngrams', 'positive_ngrams', 'negative_ngrams', 'df_clean_comments', 'positive_clean_comments' or 'negative_clean_comments'")


    #************************************ 20. CLUSTERIZE ************************************

#'distiluse-base-multilingual-cased' is the model recommended for ES by sentence_transformes

st_models = ["distiluse-base-multilingual-cased", "distiluse-base-multilingual-cased-v2",
            "paraphrase-xlm-r-multilingual-v1", "stsb-xlm-r-multilingual", "quora-distilbert-multilingual",
            "distilbert-multilingual-nli-stsb-quora-ranking", "xlm-r-100langs-bert-base-nli-stsb-mean-tokens"]

hf_models = ["bert", "distilbert"]

def clusterize(model=str, cluster_method=str, comments=np.ndarray):
    
    if model in st_models:
        to_use_vecs = st_vectorize(comments=comments, model=model)
    
    elif model in hf_models:
        to_use_vecs = hf_vectorize(comments=comments, model=model)
    
    else:
        raise ValueError("The model you selected is not an available model")

    d_metric = input("For clustering please select between the 'euclidean' or 'cosine' distance\nHint: cosine distance is preferred when texts have different length\n")
    while d_metric != "euclidean" and d_metric != "cosine":
        d_metric = input("Please select 1 of the mentioned distances\n")

    if cluster_method == "kmeans":
        
        scale = input("As you selected kMeans, choose to scale data with 'yes' or not scale with 'no':\nhint: scale is recommended with gradient descent algos like lin/log reression or distance algos like kMeans, KNN or SVM\n")
        while scale != "yes" and scale != "no":
            scale = input("Please select 'yes' or 'no'\n")
            
        #this function will recommend a number of clusters based in the KMeans algorithm
        #the bigger the score the better
        #kmax is the max number of clusters that is usually recommended
        print("----------------------------------------------------------------------------------------------------------")
        print("Getting the Shilouette Score...")  
        
        if d_metric == "cosine": 
            if scale == "yes":
                input_vecs = normalize(to_use_vecs, norm='l2')
            elif scale == "no":
                input_vecs = to_use_vecs
            #input function for tensorflow kmeans
            def input_fn():
                return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(input_vecs, dtype=tf.float32), num_epochs=1)       
            shilouette_tf(input_fn, to_use_vecs, kmax=20, distance_m=COSINE_DISTANCE, sil_metric="cosine")
            print("----------------------------------------------------------------------------------------------------------")
            k = int(input("Please write the number of clusters you want: "))
            #this function will create clusters with KMeans algorithm
            clusters = clusterK_tf(input_fn, k, distance_m=COSINE_DISTANCE) 
            
        elif d_metric == "euclidean":
            shilouette(to_use_vecs, kmax=20, scale=scale)
            print("----------------------------------------------------------------------------------------------------------")
            k = int(input("Please write the number of clusters you want: "))
            #this function will create clusters with KMeans algorithm
            clusters = clusterK(to_use_vecs, k, scale=scale)    
            
    elif cluster_method == "hdbscan":
        min_cluster_size = int(input("Please write the minimum cluster size you want\nJust take into account that usually the bigger the cluster size the less the clusters, so if your dataset is too big, it is better that you take a bigger min_cluster_size\n"))         
        clusters = clusterH(to_use_vecs, min_cluster_size=min_cluster_size, min_samples=None, distance_m=d_metric)
            
    #we will use pca algorithm to reduce the dimensions to 3, so we can visualize it
    print("----------------------------------------------------------------------------------------------------------")
    df_scaled = scaler(to_use_vecs)
    pca_3d = reducPca(df_scaled)

    print("----------------------------------------------------------------------------------------------------------")
    df = dfConcat(comments, "comments", clusters, "clusters", pca_3d, name3=["x", "y", "z"])
    
    #finally this will create a 3d visual of the clusters
    print("----------------------------------------------------------------------------------------------------------")
    print("Visualizing Clusters")
    visualize(df)
    
    #this point will create a wordcloud based in the number of clusters you choosed
    grams = input("For wordcloud choose between 'unigrams' or 'computed'\n")
    while grams != "unigrams" and grams != "computed":
        grams = input("Please select 1 of the mentioned grams\n")
        
    print("----------------------------------------------------------------------------------------------------------")
    print("Getting Wordcloud")
    for i in set(df.clusters):
        get_wordcloud(topic=None, model=None, df=df, clusters=i, grams=grams)
    
    
    if cluster_method == "kmeans":
        return df.drop(columns={"x", "y", "z"}), d_metric, scale, k, grams

    else:
        return df.drop(columns={"x", "y", "z"}), d_metric, min_cluster_size, grams


    #************************************ 20. SEARCH COMMENTS BY TOPIC ************************************

from scipy.spatial import distance


def search_by_topic(df=pd.DataFrame, top_results=int, columns=str, dist=str, vectors=np.ndarray, topics=np.ndarray):

    df_dict = {"feature":[], "id":[], "comment_type":[], "score":[], "area":[], "comment":[],
              "clean_comment":[], "language":[], "clusters":[], "distance":[], "topic":[]}

    for v, f, ids, ct, s, a, c, cc, l, cl in zip(list(df.vectors), df.feature, df.id, df.comment_type, df.score, 
                           df.area, df.comment, df.clean_comment, df.language, df.clusters):
        
        for vector, topic in zip(vectors, topics):
            
            df_dict["feature"].append(f)
            df_dict["id"].append(ids)
            df_dict["comment_type"].append(ct)
            df_dict["score"].append(s)
            df_dict["area"].append(a)
            df_dict["comment"].append(c)
            df_dict["clean_comment"].append(cc)
            df_dict["language"].append(l)
            df_dict["clusters"].append(cl)

            if dist == "cosine":
                df_dict["distance"].append(distance.cosine(vector, v))

            else:
                df_dict["distance"].append(distance.euclidean(vector, v))
                
            df_dict["topic"].append(topic)

    df_dist = pd.DataFrame(df_dict)

    if top_results > len(df_dist):
        top_results = len(df_dist)
    
    #this point group by topic, sorts by distance (nsmallest) and filter top_n results per topic
    #nsmallest(n) returns top_n smallest results, while nlargest(n) the opposite
    df_dist = df_dist.loc[df_dist.groupby('topic')['distance'].nsmallest(top_results).reset_index()['level_1']]
    
    if columns == "all":
        return df_dist.sort_values(by=["topic", "distance"]).drop_duplicates()
    
    elif columns == "specific":
        return df_dist.sort_values(by=["topic", "distance"])[["id", "area", "comment", "topic"]].drop_duplicates()
    
    else:
        raise ValueError("This function just take as input 'all' or 'comments'")


    #************************************ 19. FINAL DF SELECTION ************************************

def final_df_selection(df1=pd.DataFrame, df2=pd.DataFrame, element=str):
    if element == "df_ngrams":
        return pd.merge(df1, df2.rename(columns={"comments":"feature"}), how="left", on=["feature"])
    
    elif element == "positive_grams":
        return pd.merge(df1[df1.comment_type=="Positivo"], df2.rename(columns={"comments":"feature"}), how="left", on=["feature"])
    
    elif element == "negative_ngrams":
        return pd.merge(df1[df1.comment_type=="Negativo"], df2.rename(columns={"comments":"feature"}), how="left", on=["feature"])
    
    elif element == "df_clean_comments":
        return pd.merge(df1, df2.rename(columns={"comments":"clean_comment"}), how="left", on=["clean_comment"])
    
    elif element == "positive_clean_comments":
        return pd.merge(df1[df1.comment_type=="Positivo"], df2.rename(columns={"comments":"clean_comment"}), how="left", on=["clean_comment"])
    
    elif element == "negative_clean_comments":
        return pd.merge(df1[df1.comment_type=="Negativo"], df2.rename(columns={"comments":"clean_comment"}), how="left", on=["clean_comment"])
    
    else:
        raise ValueError("Please select 'df_ngrams', 'positive_grams', 'negative_ngrams', 'df_clean_comments', 'positive_clean_comments' or 'negative_clean_comments'")
        


    #************************************ 21. FIXED CLUSTERIZE ************************************


def clusterize_fixed(model=str, cluster_method=str, comments=np.ndarray, d_metric=None, scale=None, k=None, min_cluster_size=None, grams=None):
    
    if model in st_models:
        to_use_vecs = st_vectorize(comments=comments, model=model)
    
    elif model in hf_models:
        to_use_vecs = hf_vectorize(comments=comments, model=model)
    
    else:
        raise ValueError("The model you selected is not an available model")

    if cluster_method == "kmeans":
        
        if d_metric == "cosine": 
            if scale == "yes":
                input_vecs = normalize(to_use_vecs, norm='l2')
            elif scale == "no":
                input_vecs = to_use_vecs
            #input function for tensorflow kmeans
            def input_fn():
                return tf.compat.v1.train.limit_epochs(tf.convert_to_tensor(input_vecs, dtype=tf.float32), num_epochs=1)       
            print("----------------------------------------------------------------------------------------------------------")
            #this function will create clusters with KMeans algorithm
            clusters = clusterK_tf(input_fn, k, distance_m=COSINE_DISTANCE) 
            
        elif d_metric == "euclidean":
            print("----------------------------------------------------------------------------------------------------------")
            #this function will create clusters with KMeans algorithm
            clusters = clusterK(to_use_vecs, k, scale=scale)    
            
    elif cluster_method == "hdbscan":
        clusters = clusterH(to_use_vecs, min_cluster_size=min_cluster_size, min_samples=None, distance_m=d_metric)
            
    #we will use pca algorithm to reduce the dimensions to 3, so we can visualize it
    print("----------------------------------------------------------------------------------------------------------")
    df_scaled = scaler(to_use_vecs)
    pca_3d = reducPca(df_scaled)

    print("----------------------------------------------------------------------------------------------------------")
    df = dfConcat(comments, "comments", clusters, "clusters", pca_3d, name3=["x", "y", "z"])
    
    #finally this will create a 3d visual of the clusters
    print("----------------------------------------------------------------------------------------------------------")
    print("Visualizing Clusters")
    visualize(df)
        
    print("----------------------------------------------------------------------------------------------------------")
    print("Getting Wordcloud")
    for i in set(df.clusters):
        get_wordcloud(topic=None, model=None, df=df, clusters=i, grams=grams)
    
    
    if cluster_method == "kmeans":
        return df.drop(columns={"x", "y", "z"})

    else:
        return df.drop(columns={"x", "y", "z"})