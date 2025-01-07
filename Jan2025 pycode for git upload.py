# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 13:14:58 2025

@author: 20052
"""
import ollama
import re, time
import pandas as pd

def classify_svo_tuples_line_by_line(svo_tuples, prompt1, k=10, model1 = "gemma2:2b"):
    """
    Classifies SVO tuples using Ollama and Gemma-2b, handling lists of up to k tuples.

    Args:
        svo_tuples (list of tuples): A list of SVO tuples.
        k (int, optional): The chunk size (max no. of SVO tuples) to process at once. Defaults to 10.

    Returns:
        list: A list of classifications (A, B, or C) of length equal to the number of input
              tuples, or None if there was an issue with the LLM.
    """

    prompt = prompt1 # f'''{{\n\"{prompt1}\"\n}}'''

    for i, svo_tuple in enumerate(svo_tuples):
        prompt += f"{{SVO_TUPLE_{i+1}}}:\n" # changed the way tuples are passed
    prompt += "[END SVO TUPLES]"

    for i, svo_tuple in enumerate(svo_tuples):
         prompt = prompt.replace(f"{{SVO_TUPLE_{i+1}}}", str(svo_tuple))

    # 'gemma2:9b' 'qwen2.5:3b' 'gemma2:2b'	
    response = ollama.chat(model= model1, messages=[    
        {
            'role': 'user',
            'content': prompt,
        },
    ], stream=False)

    if response and 'message' in response and 'content' in response['message']:
        classification_lines = response['message']['content'].strip().split('\n')

        # Use a regular expression to extract only 'A', 'B', or 'C' from each line
        clean_classifications = [re.match(r"^[ABC]$", line.strip()).group(0) if re.match(r"^[ABC]$", line.strip()) else 'C' for line in classification_lines]

        # Pad with "C" if the output is less than input, to match the number of inputs
        while len(clean_classifications) < len(svo_tuples):
          clean_classifications.append('C')
        return clean_classifications[:len(svo_tuples)] # ensure output length matches input length
    else:
        return None


def classify_svo_tuples_with_variable_chunk_size(svo_tuples, prompt1, k=10, model1 = "gemma2:2b"):
    """
    Classifies SVO tuples with a variable chunk size using ollama and gemma2:9b

    Args:
      svo_tuples (list of tuples): A list of SVO tuples.
      k (int, optional): The chunk size (max no. of SVO tuples) to process at once. Defaults to 10.

    Returns:
        list: A list of classifications (A, B, or C) of length equal to the number of input
              tuples, or None if there was an issue with the LLM.

    """
    all_classifications = []
    for i in range(0, len(svo_tuples), k):
        chunk = svo_tuples[i:i+k] # create chunk of max size k
        classifications = classify_svo_tuples_line_by_line(chunk, prompt1, k, model1)
        if classifications:
          all_classifications.extend(classifications)
        else:
          print("Warning! Not able to get the classification results for this chunk:", chunk)

    return all_classifications

def classify_1_doc1(df0, prompt1, chunk_size=20, model2 = "gemma2:2b"):
    """
    Classifies SVO phrases in a DataFrame chunk-by-chunk, padding classifications if necessary.

    Args:
        df0 (pd.DataFrame): Input DataFrame with columns 'chunk_id' and 'svo_frag'.

    Returns:
        pd.DataFrame: DataFrame with an additional 'class' column containing classifications.
    """
    out_df01 = pd.DataFrame(columns=df0.columns)
    for i2 in range(df0.chunk_id.max() + 1): # +1 so that max is included in range
        df01 = df0[df0['chunk_id'] == i2]
        svo_series = df01.svo_frag
        #classifications = classify_svo_series_line_by_line(svo_series, prompt1, k=chunk_size)
        classifications = classify_svo_tuples_with_variable_chunk_size(svo_tuples_to_classify, prompt1, k=chunk_size, model1 = model2)

        if classifications is not None:
          if len(classifications) > len(svo_series):
             classifications = classifications[:len(svo_series)]
          # Pad the classifications list if it's shorter than the SVO series
          while len(classifications) < len(svo_series):
             classifications.append('C')
          df01.loc[:,'class'] = pd.Series(classifications, index=df01.index)
          out_df01 = pd.concat([out_df01, df01])
        else:
          continue
    return out_df01


# unit func
def chunkize_doc(k, prim_key0, df, column0 = 'svo_frag'):
    sub_df0 = df[df['prim_key'] == prim_key0]
    kseq_start = [] 
    for num in range(0, len(sub_df0), k):
        kseq_start.append(num)
        
    if (kseq_start[-1] < len(sub_df0)):
        kseq_start.append(len(sub_df0))
        
    df0 = pd.DataFrame(columns = ['prim_key', 'chunk_id', 'chunk'])
    
    for i0 in range(len(kseq_start)-1):
        start0 = kseq_start[i0]
        stop0 = kseq_start[i0+1]
        chunk0 = sub_df0[column0].iloc[start0:stop0]
        n1 = len(chunk0)
        df0_int = pd.DataFrame({'prim_key': [prim_key0]*n1, 'chunk_id': [i0]*n1,'chunk':chunk0})
        df0 = pd.concat([df0, df0_int])
        
    sub_df0['chunk_id'] = df0['chunk_id']
    return(sub_df0)

def classify_1_chunk(df0, chunk_size=25):
	for i2 in range(df0.chunk_id.max() + 1): # +1 so that max is included in range
		df01 = df0[df0['chunk_id'] == i2]
		svo_series = df01.svo_frag
		#classifications = classify_svo_series_line_by_line(svo_series, k=chunk_size)
		classifications = classify_svo_tuples_with_variable_chunk_size(svo_tuples_to_classify, k=chunk_size)

		if classifications is not None:
			if len(classifications) > len(svo_series):
				classifications = classifications[:len(svo_series)]

			# Pad the classifications list if it's shorter than the SVO series
			while len(classifications) < len(svo_series):
				classifications.append('C')

		df01.loc[:,'class'] = pd.Series(classifications, index=df01.index)
		out_df01 = pd.concat([out_df01, df01])
		else:
			continue
	return out_df01


def chunk_n_classify_corpus(df_corpus, chunk_size=25):
    
    df0 = pd.DataFrame(columns=df_corpus.columns)
    df0 = df0.assign(chunk_id=0)

    kseq_start = []
    for num in range(0, len(df_corpus), chunk_size):
        kseq_start.append(num)

    if (kseq_start[-1] < len(df_corpus)):
         kseq_start.append(len(df_corpus))

    for i0 in range(len(kseq_start)-1):
        start0 = kseq_start[i0]
        stop0 = kseq_start[i0+1]
        df0_int = df.iloc[start0:stop0]
        df0_int['chunk_id'] = i0
        df0 = pd.concat([df0, df0_int])
        
    return(df0)
## ========================================================================================

prompt1 = """
You are an expert in organizational behavior and linguistics. You will be given a list of subject-verb-object (SVO) tuples extracted from quarterly earnings call transcripts of publicly traded firms. Your task is to classify each tuple based on the degree of perceived organizational agency (also known as locus of control). You must explicitly analyze both the SVO order and the OVS order of the tuples.

Consider the following when classifying a tuple:

    - High Locus of Control (High LoC - A):
        - If the SUBJECT (S) is "we," "I," or any similar first-person pronoun, or a reference to the firm itself, the tuple MUST be classified as "A", as it strongly suggests the firm or its management is taking ownership and showing confidence. If the verb is a direct action verb, it will strengthen this classification.
        - If the OBJECT (O) contains "we," "I," "our," "us", or any similar first-person pronoun, or references to the firm or its management, or its teams, that tuple MUST be classified as "A". The firm showing a willingness to be associated with an action, shows high internal agency and control. The verb may or may not be active.
        - Any tuple with a subject or an object that makes a self-reference (e.g., "we", "us", "our", etc) MUST be classified as A.

    - Low Locus of Control (Low LoC - B):
        - If the SUBJECT (S) or OBJECT (O) relates to external forces such as the market, the economy, regulations, competitors, or other factors outside the firm's direct control, this indicates low agency, especially when paired with passive verbs.
        
    - Neither/Neutral (C):
      - If the SUBJECT (S) and OBJECT (O) are general descriptions, objects, or processes (without direct firm references) or if the verb is linking verb, or if neither High nor Low Locus of Control is applicable, classify it as 'C'.  If you are in doubt or cannot classify, use 'C'.

You must return the classification for each SVO tuple on a separate line. For each line, provide ONLY the category (A, B, or C), without any other text or explanation. The order of the output must correspond to the order of the input.

Here are the SVO tuples to classify:

[BEGIN SVO TUPLES])
"""
