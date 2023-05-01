# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:33:17 2023

@author: 20052
"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack, csr_matrix

# for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer
import nltk, re
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer,SnowballStemmer
stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()
stemmer  = SnowballStemmer("english")

# for model development
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#from xgboost import XGBClassifier
#from sklearn.ensemble import RandomForestClassifier

# for model evaluation
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# functionize and run wordlist based sentence extractor
def extr_wordlist_sents(doc0, wordlist0, wl_sents_num, extr_sents):
    
    if type(doc0)==float:
        doc0 = str(doc0)
        
    sent_list0 = sent_tokenize(doc0)
    keep_words0 = [] # for any doc, use only relev words
    for word0 in wordlist0:
        word1 = re.sub(r'\\\\', r'\\', word0.lower())
        if(len(re.findall(word1, doc0))>0):
            keep_words0.append(word1)

    sent_df0 = pd.Series(sent_list0); sent_df0.head()
    sents_stor0 = []

    for word0 in keep_words0:
        a0 = sent_df0.apply(lambda x: len(re.findall(word0, x)))
        a1 = [sent_df0.iloc[x] for x in range(len(a0)) if a0[x]>0]
        sents_stor0.extend(a1)
	
    sents_stor1 = list(set(sents_stor0)); sents_stor1
    wl_sents_num.append(len(sents_stor1))
    if len(sents_stor1)==0: # potential anomaly handling
        sents_stor1 = ['empty']
    sents_stor2 = " ".join(sents_stor1); sents_stor2
    extr_sents.append(sents_stor2)

    return(wl_sents_num, extr_sents)


## extract matched keyword toks and give count also
def extr_wordlist_tokens(doc0, wordlist0, num_wl_toks, extr_toks):
	num_wl_toks0 = 0
	extr_toks0 = []
	for word0 in wordlist0:
		#word1 = r'\\w*' + word0.lower() + r'\\w*'; word1
		#word2 = re.sub(r'\\\\', r'\\', word1); word2        
		a0 = re.findall(word0, doc0); a0
		num_wl_toks0 = num_wl_toks0 + len(a0)
		if len(a0) > 0:
			extr_toks0.extend(a0)

	num_wl_toks.append(num_wl_toks0)
	extr_toks.append(extr_toks0)
	return(num_wl_toks, extr_toks)

def opt_logreg_apply(dtm0, df2):

	train_x1, valid_x1, train_y1, valid_y1 = model_selection.train_test_split(dtm0, df2['relevant'], random_state=0)

	# doing grid search on available parameters
	c_params =[0.01,1,10,100] # np.linspace(0.01,1000,100)
	tuned_params = [{'C':c_params, "penalty":["l2","l1"]}]    
	lr_grid = GridSearchCV(estimator=LogisticRegression(max_iter=15000, random_state=0, solver='liblinear'),
                   param_grid = tuned_params, cv = 5, scoring = "accuracy")

	# fit the model on training dataset
	lr_grid.fit(train_x1, train_y1)  # 4 s
	print(lr_grid.best_params_)
	y_pred_valid = lr_grid.predict(valid_x1)
	print(f'Accuracy on test dataset : {round(accuracy_score(y_pred_valid,valid_y1),2)*100} %')

	# redefine opt model now
	parms_list = list(lr_grid.best_params_.values())
	model0 = LogisticRegression(max_iter=15000,random_state=0, solver='liblinear', penalty=parms_list[1], C=parms_list[0])    
	return(model0, parms_list)

# func 3: get logreg coeffs
def get_logreg_coefs(vectorizer, model0):
	feat_names = vectorizer.get_feature_names()
	coeffs = model0.coef_.tolist()[0]
	df_coef = pd.DataFrame({'token':feat_names, 'coef':coeffs}); df_coef
	df_coef1 = df_coef[df_coef['coef'] != 0]
	df_coef2 = df_coef1.sort_values(by=['coef'])
	return(df_coef2)

"""
# func 4a: tgt and extract misclassifieds. Intermed func
def misclass_inds(df, y_true, y_pred, inds): 

	misclassified = np.where(y_true != y_pred); misclassified
	a0 = np.array(misclassified).tolist(); a0[0][:8]
	a1 = [x for x in a0[0]]; len(a1)
	a2 = [inds[x] for x in a1]; a2[:8]
	df_misclassif = df.iloc[a2, :]; df_misclassif.columns
    
	y_pred_miscl = [list(y_pred)[x] for x in a1]
	df_misclassif.insert(3, "y_pred", y_pred_miscl); df_misclassif.columns	
	# df_misclassif = df_misclassif.loc[:, ['slnum', 'filename', 'sents', 'relevant', 'y_pred', 'nchar', 'cleaned_sents_1']]
	# df_misclassif['relevant'].describe()
	return(df_misclassif)

# func 4: wrapper func to extract misclassified rows
def extract_misclassifieds(df, dtm0, model0, prop1=0.33, column0='relevant'):

	# create indices for train and test
	ind_list = [x for x in range(df.shape[0])]
	# prop1 = 0.33
	n1 = int(round(prop1*len(ind_list), 0)); n1
    
	# somehow, couldn't fig out how 2 set seed for below proc
	test_inds = sample(ind_list, n1); test_inds[:8]
	train_inds = [x for x in ind_list if x not in test_inds]; train_inds[:8]

	# now split sample and run logreg again. 
	x_train = dtm0[train_inds,:]; y_train = df[column0].iloc[train_inds]
	x_test = dtm0[test_inds,:]; y_test = df[column0].iloc[test_inds]

	logreg = model0.fit(x_train, y_train)  # 0.006 s to train the model
	print("trg accu: ", logreg.score(x_train, y_train))  # 100%  on trg set, way overfitted

	y_pred_test = logreg.predict(x_test)
	y_pred_trg = logreg.predict(x_train)

	print(confusion_matrix(y_test, y_pred_test))
	print("test accu: ", logreg.score(x_test, y_test))  # 0.78 :(

	df_misclass_test = misclass_inds(df, y_test, y_pred_test, test_inds)
	df_misclass_trg = misclass_inds(df, y_train, y_pred_trg, train_inds)

	df_misclass = df_misclass_trg.append(df_misclass_test)
	return(df_misclass)
"""

def text_clean0(text):
   
    text = re.sub('<.*?>', '', str(text))    
    text = re.sub('\d+[,\.]?\d+', '', text)	
    text = re.sub('-', '_', text)
    text = re.sub('\$', 'dollar', text)
    text = re.sub('%', 'percent', text)	
    
    text = nltk.word_tokenize(str(text))
    text = [word.lower() for word in text] # lowercase text
    text = [wnl.lemmatize(i) for i in text]  # lemmatize away plurals
    
    text = [word for word in text if word not in stopword_list] # drop stopwords
    #text = [word for word in text if word.isalpha()] # drop all non-alphabetic tokens  

    return ' '.join(text)


# utility func
def npwhere2ind(list1, list2):
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
		if i0%1000 == 0:
			print(i0)
	return([out_ind, err_inds])

# sorted_ind1, err_ind1 = npwhere2ind(feat1, a2) # 10s

# func to convert huge corpus_dtm to dimns of trained model's dtm_model
from scipy.sparse import hstack, csr_matrix
def dtm_reshape(dtm_model, dtm_corpus, vect_model, vect_corpus):

	feat1 = vect_model.get_feature_names()[:dtm_model.shape[1]]; len(feat1)
	feat2 = vect_corpus.get_feature_names()[:dtm_corpus.shape[1]]; len(feat2)
	a1 = np.asarray(feat2); a1.shape
	index_overlapping, index_non_overlapping  = npwhere2ind(a1, feat1)

	new_colms = csr_matrix((dtm_corpus.shape[0], len(index_non_overlapping))); new_colms.shape
	old_colms_mat = dtm_corpus[:,index_overlapping]; old_colms_mat.shape
	# now np.hstack(x1, x2) the 2 csr matrices x1,x2
	new_csr_mat = hstack((old_colms_mat, new_colms)); new_csr_mat.shape # 505k x 27k

	# now sort colms to get same order as dtm1 tokens
	a0 = [feat1[x] for x in index_non_overlapping]; a0[:8]
	a1 = [feat2[x] for x in index_overlapping]; a1[:8]
	a2 = a1 + a0; a2[:8]
	a3 = np.asarray(a2); a3.shape
	sorted_ind, err_inds = npwhere2ind(a3, feat1)

	new_csr_mat = new_csr_mat.tocsr() # [:,sorted_ind]
	new_csr_mat = new_csr_mat[:,sorted_ind]

	return(new_csr_mat)  # whew.

"## --- func to build cleaned sents wale series for relev filtering"

def build_sents_df(doc0, prim_key0):
    sents_list0 = sent_tokenize(doc0)
    if len(sents_list0) == 0:
        sents_list0 = ['empty']
    cleaned_sents_list0 = pd.Series(sents_list0).apply(text_clean0).tolist()
    out_df10 = pd.DataFrame({'prim_key':prim_key0, 'sents': sents_list0, 'cleaned_sents': cleaned_sents_list0}) 
    return(out_df10)

'''
# out_df00 = pd.DataFrame(columns = ['prim_key', 'sents', 'cleaned_sents_1'])
def doc2sent_series(doc0, prim_key0, out_df00):
    sents_list0 = sent_tokenize(doc0); len(sents_list0) 
    # insert anomaly correction here for missing docs
    cleaned_sents0 = pd.Series(sents_list0).apply(text_clean0)
    out_df0 = pd.DataFrame({'prim_key': prim_key0, 'sents': sents_list0,
                            'cleaned_sents_1': cleaned_sents0.tolist()})
    out_df00 = pd.concat([out_df00, out_df0], axis = 0)
    return(out_df00)
'''

## Find lexical features for each doc
from lexical_diversity import lex_div as ld
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
analyzer = SentimentIntensityAnalyzer()

def build_aux_metrics(filename_series, doc_series):
	lex_vol = []; ttr = []; mtld = []; #vocd = []  # lexical div measures
	compound_mean = []; compound_std = []    
	filename = []  # sentiment measures

	for i0 in range(len(doc_series)):

		filename0 = filename_series.iloc[i0]; filename0
		doc0 = doc_series.iloc[i0]; doc0
		doc0_list = sent_tokenize(doc0); doc0_list
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

		compound_mean.append(mean_list0[3]); compound_std.append(std_list0[3])                        
		filename.append(filename0)

		flt = ld.flemmatize(doc0_string); flt
		lex_vol0 = len(flt)  # lexical volume measure
		ttr0 = ld.ttr(flt)  # basic Text-Type Ratio or TTR
		mtld0 = ld.mtld(flt) # Measure of Textual Lexical Diversity (MTLD) for lexical variability
		# vocd0 = ld.hdd(flt) # vocd or Hypergeometric distribution D (HDD), as per McCarthy and Jarvis (2007, 2010)

		lex_vol.append(lex_vol0)
		ttr.append(ttr0)
		mtld.append(mtld0)
		# vocd.append(vocd0)

		if i0%100 == 0:
			print(i0)

	# save as df
	df1 = pd.DataFrame({'prim_key':filename, 'senti_compound': compound_mean,'senti_compound_std': compound_std,
                      'lex_vol':lex_vol, 'ttr':ttr, 'mtld':mtld})
	return(df1)


# --- find readability indices for df_sents ---
import textstat
def calc_fogindex(sents_series0):
	fogIndex=[]; # flesch_kincaid=[]; flesch_readby=[];
	for i0 in range(len(sents_series0)):
		sent0 = sents_series0[i0]
		#flesch_readby.append(textstat.flesch_reading_ease(sent0))
		#flesch_kincaid.append(textstat.flesch_kincaid_grade(sent0))
		fogIndex.append(textstat.gunning_fog(sent0))
		if i0%100==0:
			print(i0)

	df_readby = pd.DataFrame({'fogIndex':fogIndex})
	return(df_readby)

