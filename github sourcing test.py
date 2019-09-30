# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:40:49 2019

@author: 20052
"""

## setup chunk
import spacy
nlp = spacy.load('en_core_web_sm')
import time
import re
import pandas as pd
import feather
from nltk.tokenize import sent_tokenize

## Routine to COUNT hypoth sents from one Doc 
def count_hypoth_oneDoc(doc1, fileName1):	

	sents_list = sent_tokenize(doc1)
	tot_sents = len(sents_list)

	# build empty list to populate as DF colms
	hypoth_sents_index = []
	hypoth_sents_counter = 0

	# t1 = time.time()
	for i1 in range(len(sents_list)):
		sent0 = sents_list[i1]
		sent0 = re.sub("\s{2,}", "", sent0)  # collapse multiple-spaces
		sent0_ann = nlp(sent0)
        
		# hypoth detection as previously
		morph0 = str([nlp.vocab.morphology.tag_map[token.tag_] for token in sent0_ann])
		if bool(re.search(r'mod', morph0))== True:
			hypoth_sents_counter = hypoth_sents_counter + 1
			hypoth_sents_index.append(i1) 
		
	# build DFs now
	doc_name = fileName1
	factual_sents_count =  tot_sents - hypoth_sents_counter 
	hypoth_sents_indices = str(set(hypoth_sents_index))
	hypoth_sent_df1 = pd.DataFrame({'docName': doc_name, 
                                  'index': hypoth_sents_indices,
                                  'fact_sents_num': factual_sents_count, 
                                  'hypoth_sents_num': hypoth_sents_counter, 
                                  'tot_sents':tot_sents }, index=[0])
	return hypoth_sent_df1

# Routine to extract hypoth sents from corpus 
def extract_hypoth(fileName, textCorpus):

	# create empty DFs to populate outp with
	outp_hypoth_sent_df = pd.DataFrame(columns = ['docName', 'index', 'fact_sents_num', 'hypoth_sents_num', 'tot_sents'])	

	# loop over files
	for i1 in range(len(fileName)):
		fileName1 = fileName.iloc[i1, 0]
		doc1 = textCorpus.iloc[i1, 0]
		try:  
			hypoth_sent_df1 = count_hypoth_oneDoc(doc1, fileName1)         
		except KeyError:
			pass
		else:
			outp_hypoth_sent_df = outp_hypoth_sent_df.append(hypoth_sent_df1)           

		if (i1%100 == 0):
			print(i1, " of ", len(fileName), "\n")
            
	return outp_hypoth_sent_df