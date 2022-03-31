# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:56:27 2022

@author: 20052
"""

# setup chunk
import spacy
from spacy import displacy
from collections import Counter
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()

## download nltk.punkt
import urllib.request
import nltk
nltk.download('punkt')


# Func 1 - routine to display sentence as DF postags
def token_attrib(sent0):
	doc = nlp(sent0)

	text=[]
	lemma=[]
	postag=[]
	depcy=[]

	for token in doc:
		text.append(token.text)
		lemma.append(token.lemma_)
		postag.append(token.pos_)
		depcy.append(token.dep_)

	test_df = pd.DataFrame({'text':text, 'lemma':lemma, 'postag':postag, 'depcy':depcy})
	return(test_df)

# test-drive above func on test data
# sent0 = "Donald Trump is a controversial American President"
# test_df = token_attrib(sent0)
# test_df

## Func 2: def func to display depcy tree for 1 sent. Note recursive struc!!
from nltk import Tree
def to_nltk_tree(node):
	if node.n_lefts + node.n_rights > 0:
		return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
	else:
		return node.orth_

# print tree for seq of sents if needed
# sent0 = "Donald Trump is a controversial American President." 
# sent = nlp(sent0)
# [to_nltk_tree(sent.root).pretty_print() for sent in sent.sents]

## Func 3: define a func to extract & display chunking ops
def chunkAttrib(sent0):

	doc = nlp(sent0)
	chunk1 = [(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text) for chunk in doc.noun_chunks]
    
	out_df1 = pd.DataFrame(chunk1, columns = ['chText', 'chRootText', 'chRootDep', 'chRootHead'])
	return(out_df1)

# test-drive above func
# sent0 = "Donald Trump is a controversial American President."
# chunk_df = chunkAttrib(sent0)  # 0.01 secs
# print(chunk_df)

