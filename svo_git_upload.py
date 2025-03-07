# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:27:40 2020

@author: 20052
"""

# https://raw.githubusercontent.com/peter3125/enhanced-subject-verb-object-extraction/master/subject_verb_object_extract.py

import spacy, nltk
import pandas as pd
import en_core_web_sm
from collections.abc import Iterable

# use spacy small model
nlp = en_core_web_sm.load()

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}


# does dependency set contain any coordinating conjunctions?
def contains_conj(depSet):
    return "and" in depSet or "or" in depSet or "nor" in depSet or \
           "but" in depSet or "yet" in depSet or "so" in depSet or "for" in depSet


# get subs joined by conjunctions
def _get_subs_from_conjunctions(subs):
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


# get objects joined by conjunctions
def _get_objs_from_conjunctions(objs):
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


# find sub dependencies
def _find_subs(tok):
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)
    return [], False


# is the tok set's left or right negated?
def _is_negated(tok):
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


# get all the verbs on tokens with negation marker
def _find_svs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# get grammatical objects for a given set of dependencies (including passive sentences)
def _get_objs_from_prepositions(deps, is_pas):
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


# get objects from the dependencies using the attribute dependency
def _get_objs_from_attrs(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# xcomp; open complement - verb has no suject
def _get_obj_from_xcomp(deps, is_pas):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


# get all functional subjects adjacent to the verb passed in
def _get_all_subs(v):
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok):
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# return the verb to the right of this verb in a CCONJ relationship if applicable
# returns a tuple, first part True|False and second part the modified verb if True
def _right_of_verb_is_conj_verb(v):
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


# get all objects for an active/passive sentence
def _get_all_objs(v, is_pas):
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    #potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    #if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs


# return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
def _is_passive(tokens):
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


# resolve a 'that' where/if appropriate
def _get_that_resolution(toks):
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return toks


# simple stemmer using lemmas
def _get_lemma(word: str):
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


# print information for displaying all kinds of things of the parse tree
def printDeps(toks):
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


# expand an obj / subj np using its chunk
def expand(item, tokens, visited):
    if item.lower_ == 'that':
        item = _get_that_resolution(tokens)

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts


# convert a list of tokens to a string
def to_str(tokens):
    if isinstance(tokens, Iterable):
        return ' '.join([item.text for item in tokens])
    else:
        return ''

# find verbs and their subjects / objects to create SVOs, detect passive/active sentences
def findSVOs(tokens):
    svos = []
    is_pas = _is_passive(tokens)
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    visited = set()  # recursion detection
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens, visited)),
                                         "!" + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                            svos.append((to_str(expand(obj, tokens, visited)),
                                         "!" + v2.lemma_ if verbNegated or objNegated else v2.lemma_, to_str(expand(sub, tokens, visited))))
                        else:
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_, to_str(expand(obj, tokens, visited))))
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "!" + v2.lower_ if verbNegated or objNegated else v2.lower_, to_str(expand(obj, tokens, visited))))
            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            svos.append((to_str(expand(obj, tokens, visited)),
                                         "!" + v.lemma_ if verbNegated or objNegated else v.lemma_, to_str(expand(sub, tokens, visited))))
                        else:
                            svos.append((to_str(expand(sub, tokens, visited)),
                                         "!" + v.lower_ if verbNegated or objNegated else v.lower_, to_str(expand(obj, tokens, visited))))
    return svos

## def funcs to extract SVOs   
def _build_svo(doc0):
	test_sent_list0 = nltk.sent_tokenize(doc0)
	test_svos1 = []
	for sent0 in test_sent_list0:
		spacy_tok0 = nlp(sent0)
		test_svos0 = findSVOs(spacy_tok0) #test_svos1 # 0s!
		test_svos1.extend(test_svos0)
	df_svos1 = pd.DataFrame(test_svos1, columns=["subj", "verb", "obj"]); df_svos1
	subj_list = [" ".join(df_svos1.subj.tolist())]
	verb_list = [" ".join(df_svos1.verb.tolist())]
	obj_list = [" ".join(df_svos1.obj.tolist())]	
	return(subj_list, verb_list, obj_list)

# %time subj_list, verb_list, obj_list = _build_svo(doc0)
def build_svo(sents_series0, filename_series0):
	test_svos = []
	subj_list=[]; verb_list=[]; obj_list=[]
	for i0 in range(len(sents_series0)):
		#filename0 = filename_series0[i0]; filename0
		doc0 = sents_series0.iloc[i0]; doc0
		subj_list0, verb_list0, obj_list0 = _build_svo(doc0)
		subj_list.append(subj_list0); subj_list
		verb_list.append(verb_list0)
		obj_list.append(obj_list0)
		if i0%5000 == 0:
			print(i0)
	df_svo0 = pd.DataFrame({'subj':subj_list, 'verb':verb_list, 'obj':obj_list})
	df_svo0.insert(0, "filename", filename_series0)
	return(df_svo0)

# test-drive abv
# %time df_svo = build_svo(df00.hyp_sents_relev, df00.filename); df_svo

def build_svo_colm(df_svo0):
	svo_colm = []
	for i0 in range(df_svo0.shape[0]):
		a0 = df_svo0.iloc[i0, :]; a0
		a1 = str(a0.subj) + str(a0.verb) + str(a0.obj); a1
		svo_colm.append(a1)
		if i0%5000 == 0:
			print(i0)

	df_svo_series = pd.DataFrame({'svo_str':svo_colm})
	return(df_svo_series)

# %time df_svo_series = build_svo_colm(df_svo)

## unit func: to make SVO sent-fragments
def _build_svo1(doc0, prim_key0):
	test_sent_list0 = nltk.sent_tokenize(doc0)
	fragments = []; sent_inds = []; sent_orig = []
    
	for i0 in range(len(test_sent_list0)):
		sent0 = test_sent_list0[i0]
		spacy_tok0 = nlp(sent0)
		test_svos0 = findSVOs(spacy_tok0) #test_svos1 # 0s!
		fragments0 = [' '.join(tup) for tup in test_svos0]
		sent_inds0 = [i0]*len(fragments0)
		sent_orig0 = [sent0]*len(fragments0)
		fragments.extend(fragments0)
		sent_inds.extend(sent_inds0)
		sent_orig.extend(sent_orig0)
        
	prim_key0_list = [prim_key0]*len(fragments)
	df_svos1 = pd.DataFrame({'prim_key': prim_key0_list, 'sent_ind':sent_inds,
                          'sent_orig': sent_orig,'svo_frag':fragments})
	return(df_svos1)

# test-drive func
#%time df_svos1 = _build_svo1(doc0, prim_key0) # 1.9s

"## --- unit func - alt ---"
def _build_svo_tuples(doc0, prim_key0):
  test_sent_list0 = nltk.sent_tokenize(doc0)
  svo_tuples = []; sent_inds = []; sent_orig = []
    
  for i0 in range(len(test_sent_list0)):
    sent0 = test_sent_list0[i0]; sent0
    spacy_tok0 = nlp(sent0)
    test_svos0 = findSVOs(spacy_tok0); test_svos0 # 0s!
    sent_inds0 = [i0]*len(test_svos0)
    sent_orig0 = [sent0]*len(test_svos0)
    svo_tuples.extend(test_svos0)
    sent_inds.extend(sent_inds0)
    sent_orig.extend(sent_orig0)
        
  prim_key0_list = [prim_key0]*len(svo_tuples)
  df_svos1 = pd.DataFrame({'prim_key': prim_key0_list, 'sent_ind':sent_inds,
                          'sent_orig': sent_orig,'svo_frag':svo_tuples})
  return(df_svos1)

## wrapper func
def build_svo1(sents_series0, filename_series0):    
	df_svos = pd.DataFrame(columns=['prim_key', 'sent_ind', 'sent_orig', 'svo_frag'])    

	for i0 in range(len(sents_series0)):
		doc0 = str(sents_series0.iloc[i0]); doc0
		prim_key0 = filename_series0.iloc[i0]
		if (doc0 == 'nan'):
			df_svos1 = pd.DataFrame({'prim_key':[prim_key0], 'sent_ind':[0], 'sent_orig':['nan'], 'svo_frag':['empty']})
		else:
			df_svos1 = _build_svo_tuples(doc0, prim_key0)

		df_svos = pd.concat([df_svos, df_svos1])
		if i0%2000 == 0:
			print(i0)
            	
	return(df_svos)

# test-drive abv
#%time df_svo_qna_run1 = build_svo1(out_df_relev3_qna.relev_txt_qna.iloc[:24000], out_df_relev3_qna.prim_key.iloc[:24000]); df_svo_qna_run1 # 1.07s per doc!
#df_svo_qna_run1.to_csv(path0_mac + "df_svo_qna_run1.csv")

## --- sample some 50k sents from QnA ---
import random
def sample_svo_frags(df, num_rows=1000, num_sents=50, text_column='svo_frag'):
    
    unique_pkey_list = df['prim_key'].unique().tolist()    
    n1 = len(unique_pkey_list); n1
    sampled_pkey_indices = random.sample(range(n1), min(num_rows, n1))
    sampled_rows = df.iloc[sampled_pkey_indices].copy() # Create copy to avoid SettingWithCopyWarning
    
    df_out = pd.DataFrame(columns=df.columns)
    for i0 in range(len(sampled_rows)):
        
        prim_key0 = sampled_rows.prim_key.iloc[i0]; prim_key0
        subdf0 = sampled_rows[sampled_rows['prim_key'] == prim_key0]; subdf0
        sampled_frag_inds0 = random.sample(range(len(subdf0)), min(num_sents, len(subdf0)))
        try:
            df_frags0 = subdf0.loc[sampled_frag_inds0]
        except KeyError:
            continue
            
        #df_frags0 = subdf0.loc[sampled_frag_inds0]                
        df_out = pd.concat([df_out, df_frags0])
        if i0%5000 == 0:
            print(i0)
    
    return df_out

# test-drive
#%time df_out_qna1 = sample_svo_frags(df_svo_qna, num_rows=1000, num_sents=100, text_column='svo_frag') # 4s
#df_out_qna1.to_csv(path0_mac + "df_out_qna1.csv")

