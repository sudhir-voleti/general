# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:15:39 2020

@author: 20052
"""

# func 1 - revise phrases & ngrams in sents
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


# func 3a - create & sample from sent-sampling frame. Unit func below
def sampl_frame(a0):
	filename0 = a0['fileName']; filename0
	doc0 = a0['sents']
	sent_list0 = nltk.sent_tokenize(doc0); sent_list0
	n1 = len(sent_list0); n1
	nchar0 = list(map(lambda x: len(x), sent_list0)); nchar0
	filename00 = [filename0]*n1; filename00
	out_df0 = pd.DataFrame({'filename': filename00, 'sents':sent_list0, 'nchar':nchar0})
	return(out_df0)

# func 3b - wrapper of sampl frame builder over a df
def build_sampl_frame(df01):    
	df0_sampl_frame = pd.DataFrame(columns = ['filename', 'sents', 'nchar'])    

	for i0 in range(df01.shape[0]):
		a0 = df01.iloc[i0,:]; a0
		out_df0 = sampl_frame(a0)  # use unit func abv
		df0_sampl_frame = df0_sampl_frame.append(out_df0)
		if i0%1000==0:
			print(i0)

	return(df0_sampl_frame)

"""
Since build_sampl_frame() repeatedly appends rows to a  DF, longer it runs, heavier the DF becomes and longer it takes

So, am breaking up the proc into steps of 10k rows each, using a small routine to help. behold.
"""

# func 3c - intermed func for start and stop points for func repeats
def start_stop_iters(df01, stepsize):
	start_list = [x for x in range(0, (df01.shape[0] - stepsize), stepsize)];  start_list
	stop_list = [x for x in range(start_list[1], df01.shape[0], stepsize)]; stop_list
	start_list.append(stop_list[len(stop_list)-1]); start_list
	stop_list.append(df01.shape[0]); stop_list
	return(start_list, stop_list)

# func 3d - iterated sampl_frame builder
def build_sampl_frame_iter(df01, stepsize):
	start_list, stop_list = start_stop_iters(df01, stepsize)
	store_list = []
	for i0 in range(len(start_list)):
		start0 = start_list[i0]; start0
		stop0 = stop_list[i0]; stop0    
		df01_sub = df01.iloc[start0:stop0,:]; df01_sub
		a00 = build_sampl_frame(df01_sub) # 50 s per 10k rows
		store_list.append(a00)
		#print("processed upto: ", stop0)

	a0 = store_list[0]; a0
	for i1 in range(1, len(store_list)):
		a0 = a0.append(store_list[i1])
	return(a0)  # df output

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
		if i0%1000 == 0:
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
	new_csr_mat = hstack((old_colms_mat, new_colms)); new_csr_mat.shape # 505k x 27k

	# now sort colms to get same order as dtm1 tokens
	a0 = [feat1[x] for x in index_non_overlapping]; a0[:8]
	a1 = [feat2[x] for x in index_overlapping]; a1[:8]
	a2 = a1 + a0; a2[:8]
	a3 = np.asarray(a2); a3.shape
	err_inds, sorted_ind = npwhere2ind(a3, feat1)

	new_csr_mat = new_csr_mat.tocsr() # [:,sorted_ind]
	new_csr_mat = new_csr_mat[:,sorted_ind]

	return(new_csr_mat.shape)  # whew.

## func 5a - unit func for summarizing relevant sents back to docs
def file2subdf(i0, df80k, df910, a1, num_keyword_sents1, sents1):
	a2 = np.where(a1 == df80k['fileName'].iloc[i0]); a2 # 0.09 s    
	a23 = re.sub(r'[\n?]','', str(a2)); a23
	a20 = re.findall('\[.+]', a23); a20

	if len(a20) == 0:
		num_keyword_sents1.append(0)
		sents0 = 'empty doc'
		sents1.append(sents0)

	else:
		a21 = str(a20[0]).strip('[]').split(","); a21
		a22 = [int(x) for x in a21]; a22
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
		if i0%1000==0:
			print(i0)

	df80k_pr = pd.DataFrame({'fileName':filename1, 'num_sents': num_sents1, 
                         'sents1':sents1, 'num_keyword_sents1':num_keyword_sents1})

	return(df80k_pr)


