# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:15:39 2020

@author: 20052
"""
import feather, re
import nltk
import pandas as pd
import numpy as np

# func 1a - decontraction
def decontracted(series0):

	# specific    
	a1 = list(map(lambda x: re.sub(r"won\'t", "will not", x), series0))
	a1 = list(map(lambda x: re.sub(r"can\'t", "can not", x), a1))    

	# general
	a1 = list(map(lambda x: re.sub(r"n\'t", " not", x), a1))    
	a1 = list(map(lambda x: re.sub(r"\'re", " are", x), a1))    
	a1 = list(map(lambda x: re.sub(r"\'s", " is", x), a1))    
	a1 = list(map(lambda x: re.sub(r"\'d", " would", x), a1))    
	a1 = list(map(lambda x: re.sub(r"\'ll", " will", x), a1))    
	a1 = list(map(lambda x: re.sub(r"\'t", " not", x), a1))    
	a1 = list(map(lambda x: re.sub(r"\'ve", " have", x), a1))    
	a1 = list(map(lambda x: re.sub(r"\'m", " am", x), a1))    

	a2 = pd.Series(a1)
	return(a2)

# func 1b - revise phrases & ngrams in sents
def cleanSeries2list(series0):
	a1 = list(map(lambda x: re.sub("forward[-\s]{1,}looking", "forward-looking", x), series0))
	a1 = list(map(lambda x: re.sub("new\s*prod", "new-prod", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*tech", "new-tech", x), a1))
	a1 = list(map(lambda x: re.sub("product\s*develop", "product-develop", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*servi", "new-servi", x), a1))
    
	a1 = list(map(lambda x: re.sub("new\s*concept", "new-concept", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*plat", "new-plat", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*capabil", "new-capabil", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*acqui", "new-acqui", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*design", "new-design", x), a1))
    
	a1 = list(map(lambda x: re.sub("new\s*featu", "new-featu", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*innov", "new-innov", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*launch", "new-launch", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*oppor", "new-oppor", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*brand", "new-brand", x), a1))
	a1 = list(map(lambda x: re.sub("new\s*initi", "new-initi", x), a1))
    
	a2 = pd.Series(a1)
	return(a2)

# func 2a - unit func for encoding key-phrases into unigrams at doc level
def resub_phrase(firstw, lastw, doc0):
	refindterm0 = firstw + "\s\w+?\s" + lastw; refindterm0

	a0 = re.findall(refindterm0, doc0); a0
	n1 = len(a0)
	if n1 > 1:

		for i0 in range(n1):
			a00 = a0[i0]
			a1=str(a00).strip('[]').split(' '); a1
			regterm0 = firstw + ' ' + a1[1] + ' ' + lastw;regterm0
			regterm1 = firstw + '-' + a1[1] + '-' + lastw;regterm1	
			doc0 = re.sub(regterm0, regterm1, doc0); doc0

	return(doc0)

# func 2b - corpus level wrapper on abv unit func
def keyphrase_resub(series0):
    
	a1 = list(map(lambda x: resub_phrase('new', 'prod', x), series0))
	a1 = list(map(lambda x: resub_phrase('new', 'devel', x), a1))
	a1 = list(map(lambda x: resub_phrase('new', 'tech', x), a1))
	a1 = list(map(lambda x: resub_phrase('new', 'research', x), a1))
        
	a2 = pd.Series(a1)
	return(a2)

# func 2c - text -cleaning
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer,SnowballStemmer
stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()
def text_clean0(text, lemmatize=1, stopwords=1):  # lemmatize = 0 by default
   
	text = re.sub('<.*?>', '', str(text))    
	text = re.sub('\d+[,\.]?\d+', '', text)	
	text = re.sub('-', '_', text)
	text = re.sub('\$', 'dollar', text)
	text = re.sub('%', 'percent', text)	
    
	text = word_tokenize(str(text))
	text = [word.lower() for word in text] # lowercase text
	
	if lemmatize == 1:
		text = [wnl.lemmatize(i) for i in text]  # lemmatize away plurals
    
	if stopwords == 1:
		text = [word for word in text if word not in stopword_list] # drop stopwords

	#text = [word for word in text if word.isalpha()] # drop all non-alphabetic tokens  

	return ' '.join(text)

# a0 = list(map(lambda x: text_clean0(x, 0, 0), df01['sents'])) # 0.6 s


# func 3a - create & sample from sent-sampling frame. Unit func below
def sampl_frame(filename_series0, sents_series0):
	filename0 = filename_series0; filename0
	doc0 = sents_series0; doc0
	sent_list0 = sent_tokenize(doc0); sent_list0
	n1 = len(sent_list0); n1
	nchar0 = list(map(lambda x: len(x), sent_list0)); nchar0
	filename00 = [filename0]*n1; filename00
	out_df0 = pd.DataFrame({'filename': filename00, 'sents':sent_list0, 'nchar':nchar0})
	return(out_df0)


# func 3b - wrapper of sampl frame builder over a df
def build_sampl_frame(filename_series0, sents_series0):    
	df0_sampl_frame = pd.DataFrame(columns = ['filename', 'sents', 'nchar'])    
	for i0 in range(len(filename_series0)):
		# a0 = df01.iloc[i0,:]; a0
		filename0 = filename_series0.iloc[i0]; filename0
		doc0 = sents_series0.iloc[i0]; doc0
		out_df0 = sampl_frame(filename0, doc0)  # use unit func abv
		df0_sampl_frame = df0_sampl_frame.append(out_df0)
		if i0%5000==0:
			print(i0)
	return(df0_sampl_frame)


"""
Since build_sampl_frame() repeatedly appends rows to a  DF, longer it runs, heavier the DF becomes and longer it takes
So, am breaking up the proc into steps of 10k rows each, using a small routine to help. behold.
"""

# func 3c - intermed func for start and stop points for func repeats
def start_stop_iters(filename_series0, stepsize):
	start_list = [x for x in range(0, (len(filename_series0) - stepsize), stepsize)];  start_list
	stop_list = [x for x in range(start_list[1], len(filename_series0), stepsize)]; stop_list
	start_list.append(stop_list[len(stop_list)-1]); start_list
	stop_list.append(len(filename_series0)); stop_list
	return(start_list, stop_list)

# func 3d - iterated sampl_frame builder
def build_sampl_frame_iter(filename_series0, sents_series0, stepsize): 
	start_list, stop_list = start_stop_iters(filename_series0, stepsize)
	store_list = []
	for i0 in range(len(start_list)):
		start0 = start_list[i0]; start0
		stop0 = stop_list[i0]; stop0    
		file_sub = filename_series0.iloc[start0:stop0]
		sents_sub = sents_series0.iloc[start0:stop0]
		a00 = build_sampl_frame(file_sub, sents_sub) # 50 s per 10k rows
		store_list.append(a00)
		print("processed upto: ", stop0)

	a0 = store_list[0]; a0
	for i1 in range(1, len(store_list)):
		a0 = a0.append(store_list[i1])
	return(a0)  # df output

#%time df_sents = build_sampl_frame_iter(df01.fileName, df01.sents1, 5000) # 9.4s for 5k rows

# func 4a utility func to using numpy's fast lookup
def npwhere2ind(list1, list2): # list1 is large list from whch 2 lookup, list2 smaller one
	a3 = np.asarray(list1); a3.shape
	out_ind = []
	err_inds = []
	for i0 in range(len(list2)):
		a00 = list2[i0]; a00
		a20 = np.where(a3 == a00); a20
		a21 = re.findall('\[\d+]', str(a20)); a21
		if len(a21) == 0:
			err_inds.append(i0)
			continue
		a22 = int(str(a21[0]).strip('[]')); a22
		out_ind.append(a22)
		if i0%10000 == 0:
			print(i0)
	return([out_ind, err_inds])

# %time sorted_ind1, err_ind1 = npwhere2ind(feat1, a2) # 10s

# func 4b to convert huge corpus_dtm to dimns of trained model's dtm_model
from scipy.sparse import hstack
def dtm_reshape(dtm_model, dtm_corpus, vect_model, vect_corpus):

	feat1 = vect_model.get_feature_names()[:dtm_model.shape[1]]; len(feat1)
	feat2 = vect_corpus.get_feature_names()[:dtm_corpus.shape[1]]; len(feat2)
	a1 = np.asarray(feat2); a1.shape
	index_overlapping, index_non_overlapping  = npwhere2ind(a1, feat1)

	new_colms = csr_matrix((dtm_corpus.shape[0], len(index_non_overlapping))); new_colms.shape
	old_colms_mat = dtm_corpus[:,index_overlapping]; old_colms_mat.shape
	# now np.hstack(x1, x2) the 2 csr matrices x1,x2
	new_csr_mat = hstack((old_colms_mat, new_colms))
	print(new_csr_mat.shape) # 505k x 27k

	# now sort colms to get same order as dtm1 tokens
	a0 = [feat1[x] for x in index_non_overlapping]; a0[:8]
	a1 = [feat2[x] for x in index_overlapping]; a1[:8]
	a2 = a1 + a0; a2[:8]
	a3 = np.asarray(a2); a3.shape
	sorted_ind, err_inds = npwhere2ind(a3, feat1)

	new_csr_mat = new_csr_mat.tocsr() # [:,sorted_ind]
	new_csr_mat = new_csr_mat[:,sorted_ind]
	print(new_csr_mat.shape)
	return(new_csr_mat)  # whew.


## func 5a - unit func for summarizing relevant sents back to docs
def file2subdf(i0, df80k, df910, a1, num_keyword_sents1, sents1):
	a2 = np.where(a1 == df80k['fileName'].iloc[i0]); a2 # 0.09 s    
	#a23 = re.sub(r'[\n?]','', str(a2)); a23
	#a20 = re.findall('\[.+]', a23); a20
	a23 = np.array(a2).tolist(); a20 = a23[0]; a20

	if len(a20) == 0:
		num_keyword_sents1.append(0)
		sents0 = 'empty doc'
		sents1.append(sents0)

	else:
		#a21 = str(a20[0]).strip('[]').split(","); a21
		a22 = a20 # [int(x) for x in a20]; a22
		df_sub0 = df910.iloc[a22,:]; df_sub0
		df_sub1 = df_sub0[df_sub0['relevant']==1]; df_sub1
		num_keyword_sents1.append(df_sub1.shape[0]); num_keyword_sents1
		sents0 = ' '.join(df_sub1['sents'].tolist()); sents0
		sents1.append(sents0)

	return(num_keyword_sents1, sents1)

## func 5b - wrapper func for above
def sent2doc(df80k, df910):
	num_keyword_sents1 = []; sents1 = []; num_sents1 = []; filename1 = []
	docs = list(df910['filename']); len(docs)
	a1 = np.asarray(docs)
	for i0 in range(df80k.shape[0]):
		filename1.append(df80k['fileName'].iloc[i0])
		num_sents1.append(df80k['num_sents'].iloc[i0])
		num_keyword_sents1, sents1 = file2subdf(i0, df80k, df910, a1, num_keyword_sents1, sents1)
		if i0%10000==0:
			print(i0)

	df80k_pr = pd.DataFrame({'fileName':filename1, 'num_sents': num_sents1, 
                         'sents1':sents1, 'num_keyword_sents1':num_keyword_sents1})

	return(df80k_pr)

"""
Below requires that keyword_list be pre-specified. Typically iteratively done.
"""

# func 6a - intermed func for kryword detection in sampled sents. 
def detect_keywrds(keyword_stems, text0):
	keywrd_list = []
	for i0 in range(len(keyword_stems)):
		regex = '\\b' + keyword_stems[i0] + '\w*'; regex
		a0 = re.findall(regex, text0); a0
		if len(a0) > 0:
			# keywrd_list.append(str(a0).strip('[]').strip('\''))	
			keywrd_list.extend(a0)	
	return(keywrd_list)

# test-drive
#text0 = "innovative technology and patent based solutions are absolutely what we do uniquely well."; text0
#%time a1 = detect_keywrds(keyword_list, text0); a1

## func 6b: wrapper func. 
def extract_keyword_compts(series0, keyword_stems):

	keywrds_colm = []; keywrds_num = []; 

	for i1 in range(series0.shape[0]):
		text0 = series0.iloc[i1]
		if type(text0) != str:
			text0 = 'empty row'

		a1 = detect_keywrds(keyword_stems, text0.lower()); a1 # list
		a2 = list(set(a1))
		a2.sort()  # sort list elems alphabetically

		keywrds_colm.append(a2)
		keywrds_num.append(len(a2))

		if i1%1000 == 0:
			print(i1)

	print(len(keywrds_colm), len(keywrds_num), series0.shape[0])
	df_out = pd.DataFrame({'keywords':keywrds_colm, 'num_keywords':keywrds_num})
	return(df_out)

# test-drive
#%time df_out = extract_keyword_compts(df01['sents'], keyword_list) # 1.6s
#df01 = df01.drop(['keywords', 'num_keywords'], axis = 1); df01.columns
#df01.insert(4, 'keywords', df_out['keywords']); df01.columns
#df01.insert(5, 'num_keywords', df_out['num_keywords']); df01.columns

## for model development
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


# for model evaluation
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold

# Hyper parameter tuning
from sklearn.model_selection import GridSearchCV
import pickle

# func 2b: run ML model
def opt_logreg_apply(dtm0, yseries0, cv1=5):

	train_x1, valid_x1, train_y1, valid_y1 = model_selection.train_test_split(dtm0, yseries0, random_state=0)

	# in the following code we will do the grid search on available parameters
	c_params =[0.01,1,10,100] # np.linspace(0.01,1000,100)
	tuned_params = [{'C':c_params, "penalty":["l2","l1"]}]    
	lr_grid = GridSearchCV(estimator=LogisticRegression(max_iter=15000, random_state=0, solver='liblinear'),
                   param_grid = tuned_params, cv = cv1, scoring = "accuracy")

	# lets fit the model on training dataset
	lr_grid.fit(train_x1, train_y1)  # 4 s
	print(lr_grid.best_params_)
	y_pred_valid = lr_grid.predict(valid_x1)
	print(f'Accuracy on test dataset : {round(accuracy_score(y_pred_valid,valid_y1),2)*100} %')

	# redefine opt model now
	parms_list = list(lr_grid.best_params_.values())
	model0 = LogisticRegression(max_iter=15000,random_state=0, solver='liblinear', penalty=parms_list[1], C=parms_list[0])
	model0.fit(train_x1, train_y1)  # imp step b4 outputting model0  
	return(model0, parms_list)

# %time model0, parms_list = opt_logreg_apply(dtm_tf, df01['dem'])

## Find lexical features for each doc
from lexical_diversity import lex_div as ld
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
analyzer = SentimentIntensityAnalyzer()

def build_aux_metrics(filename_series, doc_series):
	lex_vol = []; ttr = []; mtld = []; vocd = []  # lexical div measures
	neg_mean = []; neu_mean = []; pos_mean = []; compound_mean = []
	neg_std = []; neu_std = []; pos_std = []; compound_std = []    
	filename = []  # sentiment measures

	for i0 in range(len(doc_series)):

		filename0 = filename_series.iloc[i0]; filename0
		doc0 = doc_series.iloc[i0]; doc0
		doc0_list = nltk.sent_tokenize(doc0); doc0_list
		doc0_string = " ".join(doc0_list); doc0_string
		n1 = len(doc0_list); n1

		if n1 > 1:
			vs_list = []	
			for i1 in range(n1):
				sent0 = doc0_list[i1]
				vs0 = analyzer.polarity_scores(sent0); vs0
				vs_list.append(vs0)
	
			doc0_df = pd.DataFrame(vs_list); doc0_df	
			mean_list0 = [x for x in doc0_df.mean()]; mean_list0
			std_list0 = [x for x in doc0_df.std()]; std_list0

		else:
			mean_list0 = [float(0) for x in range(4)]; mean_list0
			std_list0 = [float(0) for x in range(4)]; std_list0

		neg_mean.append(mean_list0[0]); neu_mean.append(mean_list0[1])
		pos_mean.append(mean_list0[2]); compound_mean.append(mean_list0[3])                        		
		neg_std.append(std_list0[0]); neu_std.append(std_list0[1])
		pos_std.append(std_list0[2]); compound_std.append(std_list0[3])                        
		filename.append(filename0)

		flt = ld.flemmatize(doc0_string); flt
		lex_vol0 = len(flt)  # lexical volume measure
		ttr0 = ld.ttr(flt)  # basic Text-Type Ratio or TTR
		mtld0 = ld.mtld(flt) # Measure of Textual Lexical Diversity (MTLD) for lexical variability
		vocd0 = ld.hdd(flt) # vocd or Hypergeometric distribution D (HDD), as per McCarthy and Jarvis (2007, 2010)

		lex_vol.append(lex_vol0)
		ttr.append(ttr0)
		mtld.append(mtld0)
		vocd.append(vocd0)

		if i0%5000 == 0:
			print(i0)

	# save as df
	df1 = pd.DataFrame({'filename':filename, 
                     'senti_neg': neg_mean, 'senti_neu': neu_mean, 'senti_pos': pos_mean, 'senti_compound': compound_mean,
                     'senti_neg_std': neg_std, 'senti_neu_std': neu_std, 'senti_pos_std': pos_std, 'senti_compound_std': compound_std,
                      'lex_vol':lex_vol, 'ttr':ttr, 'mtld':mtld, 'vocd':vocd})
	return(df1)

# %time df_senti = build_aux_metrics(df80k['fileName'], df80k['sents']) # 7 min

# --- find rewadability indices for df_sents ---
import textstat
def calc_readby(sents_series0):
	fogIndex=[]; flesch_kincaid=[]; flesch_readby=[];
	for i0 in range(len(sents_series0)):
		sent0 = sents_series0[i0]
		flesch_readby.append(textstat.flesch_reading_ease(sent0))
		flesch_kincaid.append(textstat.flesch_kincaid_grade(sent0))
		fogIndex.append(textstat.gunning_fog(sent0))
		if i0%10000==0:
			print(i0)

	df_readby = pd.DataFrame({'flesch_readby':flesch_readby, 'flesch_kincaid':flesch_kincaid, 'fogIndex':fogIndex})
	return(df_readby)

# %time calc_readby(df_merged2.sents1[:10000])

## --- try basic wordcl plotting in py
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def build_wordcl(text_series0):
	text = " ".join(review for review in text_series0) # 0.07s

	# Create stopword list:
	stopwords = set(STOPWORDS)
	#stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

	# Generate a word cloud image
	wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text) # 9.1s

	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.show()

#%time build_wordcl(df_dem_sents.cleaned_sents_2) # 9.8s

# func 2 build adj_matrix for dendogms and cogs
def dtm2adjacen(dtm, tf_vect, cutoff=200):

	# make adjacency mat outta dtm
	adjacen0 = dtm.T*dtm; adjacen0.shape # 2.5s
	a0 = adjacen0.sum(axis=0).tolist(); len(a0[0]) # 0.03s
	colsums0 = [int(elem) for elem in a0[0]]; colsums0[:8] # 0.1s

	# sort according to colsums
	ind0 = np.argsort(np.array(colsums0))[::-1].tolist(); ind0[:8] # 0.01s
	a0 = adjacen0[:,ind0]; a0.shape
	a1 = a0[ind0,:]; a1.shape
	a2 = a1.toarray(); a2.shape
	np.fill_diagonal(a2, 0); a2[:8,:8]  # make diags zero

	# get feature names	
	feat1 = tf_vect.get_feature_names(); len(feat1)
	colnames0 = [feat1[x] for x in ind0[:cutoff]]; colnames0[:8]

	# build DF around the sorted array
	adjacen1 = pd.DataFrame(a2[:cutoff,:cutoff]); adjacen1.iloc[:8,:8]
	adjacen1.columns = colnames0
	adjacen1.index = colnames0	
	return(adjacen1)

## home brewing a cosine simil func​
import numpy as np
def cos_simil(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    numer = np.dot(vec_a, vec_b)
    abs_vec_a = sum(vec_a*vec_a)**0.5
    abs_vec_b = sum(vec_b*vec_b)**0.5
    denom = abs_vec_a *abs_vec_b

    cos1 = numer/denom
    return(cos1)

# doc2vec model
def simil_corpus(model, dem_stmt1, k = len(model.docvecs)):

	test_doc_tokenized = word_tokenize(dem_stmt1.lower()); test_doc_tokenized
	v1 = model.infer_vector(test_doc_tokenized); v1
	# %time a0 = cos_simil(v1, model.docvecs[1]); a0 # 0.015s

	simil_scores1 = []
	for i0 in range(k): # len(model.docvecs)
		simil0 = cos_simil(v1, model.docvecs[i0])
		simil_scores1.append(simil0)
		if i0%5000 == 0:
			print(i0)

	return(simil_scores1)

# %time simil_list1 = simil_corpus(model, dem_stmt1) # 9.9s	

## define py func to refine DTMs by top n tokens
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def series2dtm(series0, min_df1=5, ngram_range1=(1,2), top_n=200):

	# build TF wala dtm
	tf_vect = CountVectorizer(lowercase=False, min_df=min_df1, ngram_range=ngram_range1)
	dtm_tf = tf_vect.fit_transform(series0)

	# refine and dimn-reduce dtm to top 10% (say) terms
	pd0 = pd.Series(dtm_tf.sum(axis=0).tolist()[0])
	ind0 = pd0.sort_values(ascending=False).index.tolist()[:top_n]
	feat0 = pd.Series(tf_vect.get_feature_names()).iloc[ind0]
	dtm_tf1 = dtm_tf[:,ind0].todense()
	dtm_df = pd.DataFrame(data=dtm_tf1, columns=feat0.tolist())

	return(dtm_df)

# test-drive abv
# dtm_dem = series2dtm(df00.cleaned_sents_2.iloc[dem_only]) # 6s
