# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:00:28 2020

@author: 20052
"""
# setup chunk
import re, time
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim setup
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# import gensim.parsing.preprocessing
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_numeric
from nltk.stem.wordnet import WordNetLemmatizer   
from nltk.corpus import stopwords
import string  # for the .join() func
import matplotlib.pyplot as plt

lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')

## routine 1 - textclean per doc
def textClean(corpus_raw):
    text1 = [strip_punctuation(doc) for doc in corpus_raw]
    text1 = [strip_tags(doc) for doc in text1]
    text1 = [strip_numeric(doc) for doc in text1]
    text1 = [[" ".join([i for i in doc.lower().split() if i not in stop_words])] for doc in text1]
    text2 = [[word for word in ' '.join(doc).split()] for doc in text1]
    normalized = [[" ".join([lemma.lemmatize(word) for word in ' '.join(doc).split()])] for doc in text1]
    return normalized

## routine 2 - gridsearch on coherence vals
def compute_coherence_values1(dictionary, corpus, texts, id2word, num_topics_list):
    coherence_values = []
    model_list = []
    #num_topics1 = [i for i in range(start, limit, step)]
    for num_topics in num_topics_list:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
                                           update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values  # note, list of 2 objs returned

## routine 2a - plot coherence metrics. input is outp of prev func
import matplotlib.pyplot as plt
def plot_coherence(coherence_values, num_topics_list):

	coher = [(coherence_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
	opt_num_topics = [y for (x,y) in coher if x == max(coherence_values)]

	x = num_topics_list
	plt.plot(x, coherence_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Coherence score")
	plt.axvline(x = opt_num_topics, color='r', label='opt_numTopics')
	plt.legend(("coherence_values"), loc='best')
	plt.show()

## routine 3 - gridsearch via perplexity scores
def compute_coherence_values1(dictionary, corpus, texts, num_topics_list):
    coherence_values = []
    model_list = []
    #num_topics1 = [i for i in range(start, limit, step)]
    for num_topics in num_topics_list:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
                                           update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence()); print(num_topics)

    return model_list, coherence_values  # note, list of 2 objs returned

## routine 3a - plot perplexity metrics
def plot_perplexity(perplexity_values, num_topics_list):

	perpl = [(perplexity_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
	opt_num_topics = [y for (x,y) in perpl if x == min(perplexity_values)]

	plt.plot(num_topics_list, perplexity_values)
	plt.xlabel("Num Topics")
	plt.ylabel("Perplexity score")
	plt.legend(("perplexity_values"), loc='best')
	plt.axvline(x = opt_num_topics, color='r')
	plt.show()
    
## routine 4 - get factor matrices
def build_beta_df(lda_model, id2word):  # lda_model is the optimal_model here
    beta = lda_model.get_topics()  # shape (num_topics, vocabulary_size).
    beta_df = pd.DataFrame(data=beta)

    # convert colnames in beta_df 2 tokens
    token2col = list(id2word.token2id)
    beta_df.columns = token2col
    # beta_df.loc[0,:].sum()  # checking if rows sum to 1

    # convert rownames too, eh? Using format(), .shape[] and range()
    rowNames=['topic' + format(x+1, '02d') for x in range(beta_df.shape[0])]
    rowNames_series = pd.Series(rowNames)
    beta_df.rename(index=rowNames_series, inplace=True)
    return(beta_df)

## routine 4a - get gamma matrix
def build_gamma_df(lda_model, corpus_raw):  # lda_model is the optimal_model here
    gamma_doc = []  # empty list 2 populate with gamma colms
    num_topics = lda_model.get_topics().shape[0]
    
    for doc in range(len(corpus_raw)):
        doc1 = str(corpus_raw[doc]).split()
        bow_doc = id2word.doc2bow(doc1)
        gamma_doc0 = [0]*num_topics  # define list of zeroes num_topics long
        gamma_doc1 = lda_model.get_document_topics(bow_doc)
        gamma_doc2_x = [x for (x,y) in gamma_doc1]#; gamma_doc2_x
        gamma_doc2_y = [y for (x,y) in gamma_doc1]#; gamma_doc2_y
        for i in range(len(gamma_doc1)):
            x = gamma_doc2_x[i]
            y = gamma_doc2_y[i]
            gamma_doc0[x] = y  # wasn't geting this in list comprehension somehow 
        gamma_doc.append(gamma_doc0)
        
    gamma_df = pd.DataFrame(data=gamma_doc)  # shape=num_docs x num_topics
    topicNames=['topic' + format(x+1, '02d') for x in range(num_topics)]
    topicNames_series = pd.Series(topicNames)
    gamma_df.rename(columns=topicNames_series, inplace=True)
    return(gamma_df)    

## routine 5 - get dominant Topic DF
def domi_topic_df(gamma_df):
	row0 = gamma_df.values.tolist()
	row=[]
	for i in range(len(row0)):
		row1 = list(enumerate(row0[i]))
		row1_y = [y for (x,y) in row1]
		max_propn = sorted(row1_y, reverse=True)[0]
		row2 = [(i, x, y) for (x, y) in row1 if y==max_propn]
		row.append(row2)

	sent_topics_df = pd.DataFrame()
	for row1 in row:
		for (doc_num, topic_num, prop_topic) in row1:
			wp = optimal_model.show_topic(topic_num)
			topic_keywords = ", ".join([word for word, prop in wp])
			sent_topics_df = sent_topics_df.append(pd.Series([int(doc_num), int(topic_num), 
                                                          round(prop_topic,4), 
                                                          topic_keywords]), 
                                                       ignore_index=True)
    
	sent_topics_df.columns = ['Doc_num', 'Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
	return(sent_topics_df)    

## Routine 6 - processing raw data
def build_gensim_corpus(corpus_raw):
    corpus_cleaned = textClean(corpus_raw)  # corpus cleaned of html tags, puncs, lemmas
    corpus_tokenized = [[word for word in ' '.join(doc).split()] for doc in corpus_cleaned]  # word_tokenize first
    id2word = corpora.Dictionary(corpus_tokenized)  # Create Dictionary from word_tokenized corpus
    corpus_gensim = [id2word.doc2bow(text) for text in corpus_tokenized]  # Building gensim corpus. TF DTM creation.
    return(corpus_cleaned, corpus_tokenized, id2word, corpus_gensim)


## Routine 7 - get factor matrices using optimal K
def ltm_outp_df(model_list, num_topics_list, id2word, K):
    K1 = K  - num_topics_list[0]; K1   # account for starting point offset
    optimal_model = model_list[K1]
    # model_topics = optimal_model.show_topics(formatted=False)

    beta_df = build_beta_df(optimal_model, id2word)  # 0.004 secs
    beta_df = beta_df.T

    gamma_df = build_gamma_df(optimal_model, corpus_cleaned); gamma_df.shape # gamma_df.iloc[:8,:8]
    sent_topics_df = domi_topic_df(gamma_df)  # 2.64 secs

    return(beta_df, gamma_df, sent_topics_df)

## Routine 8 - wrapper over all above funcs
def ltm_wrapper(corpus_raw, num_topics_list):  # start1, limit1, step1
    
    corpus_cleaned, corpus_tokenized, id2word, corpus_gensim = build_gensim_corpus(corpus_raw) 
    print("build_gensim_corpus done.\n")
    # num_topics_list = [x for x in range(start1, limit1, step1)]; num_topics_list
    
    model_list, coherence_values = compute_coherence_values1(id2word, corpus_gensim, corpus_tokenized, num_topics_list)    
       
    perplexity_values = compute_perplexity_values(model_list, corpus_gensim, num_topics_list)
    
    print("grid searches done.\n")
    
    # print gridSearch results
    coher = [(coherence_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
    perpl = [(perplexity_values[i0], num_topics_list[i0]) for i0 in range(len(num_topics_list))]
    opt_num_topics_coher = [y for (x,y) in coher if x == max(coherence_values)]; opt_num_topics_coher[0]
    opt_num_topics_perpl = [y for (x,y) in perpl if x == min(perplexity_values)]; opt_num_topics_perpl[0]
	
    # display plots		
    plot_coherence(coherence_values, num_topics_list)
    plot_coherence(coherence_values, num_topics_list)
		
    print("opt_num_topics_coher: ", opt_num_topics_coher[0])
    print("opt_num_topics_perpl: ", opt_num_topics_perpl[0])
    
    K = opt_num_topics_coher[0]; print("optimal num_topix: ", K,"\n")  # default
    
    K1 = K  - num_topics_list[0]; print("K1: ", K1, "\n")   # account for starting point offset
    optimal_model = model_list[K1]
    
    beta_df = build_beta_df(optimal_model, id2word)  # 0.004 secs
    beta_df = beta_df.T; beta_df.shape
    
    gamma_df = build_gamma_df(optimal_model, corpus_cleaned); gamma_df.shape 
    sent_topics_df = domi_topic_df(gamma_df)  
    print("factor matrices done.\n")
    
    return(beta_df, gamma_df, sent_topics_df)

