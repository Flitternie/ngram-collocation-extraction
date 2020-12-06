#!/bin/env python

import argparse
import pandas as pd
import nltk
import json
import conllu as cl

punct = "+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）《》“”()»〔〕-]+"

def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    corpus = []
    for sentence in data:
        corpus.append([(token['form'], token['upos']) for token in sentence])
    unlist_corpus = [token for sent in corpus for token in sent]
    return unlist_corpus

def ngrams(corpus, n):
    assert n==2 or n==3, "Only Bigram and Trigram are supported."
    if n==2:
        ngrams = nltk.collocations.BigramAssocMeasures()
        ngram_finder = nltk.collocations.BigramCollocationFinder.from_words(corpus)
    elif n==3:
        ngrams = nltk.collocations.TrigramAssocMeasures() 
        ngram_finder = nltk.collocations.TrigramCollocationFinder.from_words(corpus)
    return ngrams, ngram_finder

def check_punct(pair):
    for token in pair:
        if token[0] in punct:
            return False
    return True
    
def remove_punct(df):
#    df = df[df['ngram'].map(check_punct)]
    return df
        
def extract_collocation(ngrams, ngram_finder, k=10):
    freq = ngram_finder.ngram_fd.items()
    freq_table = pd.DataFrame(list(freq), columns=['ngram','freq']).sort_values(by='freq', ascending=False)
    freq_table = remove_punct(freq_table)
    freq_top = freq_table[:k].ngram.values
    freq_top_score = freq_table[:k]['freq'].values
    
    #PMI
    pmi_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.pmi)), columns=['ngram','PMI']).sort_values(by='PMI', ascending=False)
    pmi_table = remove_punct(pmi_table)
    pmi_top = pmi_table[:k].ngram.values
    pmi_top_score = pmi_table[:k]['PMI'].values
    
    #t-test
    t_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.student_t)), columns=['ngram','t']).sort_values(by='t', ascending=False)
    t_table = remove_punct(t_table)
    t_top = t_table[:k].ngram.values
    t_top_score = t_table[:k]['t'].values
    
    #chi-square
    chi_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.chi_sq)), columns=['ngram','chi-sq']).sort_values(by='chi-sq', ascending=False)
    chi_table = remove_punct(chi_table)
    chi_top = chi_table[:k].ngram.values
    chi_top_score = chi_table[:k]['chi-sq'].values
    
    #likelihood
    lik_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.likelihood_ratio)), columns=['ngram','likelihood']).sort_values(by='likelihood', ascending=False)
    lik_table = remove_punct(lik_table)
    lik_top = lik_table[:k].ngram.values
    lik_top_score = lik_table[:k]['likelihood'].values
    
    ngrams_compare = pd.DataFrame([freq_top, freq_top_score, pmi_top, pmi_top_score, t_top, t_top_score, chi_top, chi_top_score, lik_top, lik_top_score]).T
    ngrams_compare.columns = ['Frequency','Score', 'PMI','Score', 'T-test','Score', 'Chi-Sq Test','Score', 'Likeihood Ratio Test', 'Score']
    return ngrams_compare

def load_check_data(check_file):
    with open(check_file, 'r', encoding='utf-8') as cf:
        conllulist = []
        for tokenlist in cl.parse_incr(cf):
            conllulist.append(tokenlist) 
    sentlist = [[token['form'] for token in sent] for sent in conllulist]
    return conllulist, sentlist
    
def check_collocation(col, conllulist, sentlist):
    result = {}
    for word_rank, word in enumerate(col):
        assert word_rank == 0 or word_rank == 1
        for sent, conllu in zip(sentlist, conllulist):
            if word[0] in sent:
                idx = sent.index(word[0])
                if conllu[idx]['upos'] != word[1]:
                    continue
                pair_idx = int(conllu[idx]['head'])-1
                if (conllu[pair_idx]['form'], conllu[pair_idx]['upos']) == col[abs(word_rank-1)] and abs(int(conllu[pair_idx]['id'])-int(conllu[idx]['id'])) == 1:
                    if conllu[idx]['deprel'] in result.keys():
                        result[conllu[idx]['deprel']] += 1 
                    else:
                        result[conllu[idx]['deprel']] = 1 
    return result if result != {} else "No Relation Found"
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="collocation.py")
#    parser.add_argument('-corpus', required=True, type=str,
#                        help="path to the corpus")
#    parser.add_argument('-conllu', default=None, type=str,
#                        help="path to the CoNLL-U file for collocation checking")
#    parser.add_argument('-n', required=True, default=2, type=int,
#                        help="ngram collocations to be extracted")
#    parser.add_argument('-k', default=10, type=int,
#                        help="top k collocations to be compared and output")

    args = parser.parse_args()
    args.corpus='./corpus.json'
    args.conllu='ud.conllu'
    args.n=2
    args.k=20
    
    corpus = load_data(args.corpus)
    ngrams, ngram_finder = ngrams(corpus, args.n)
    ngrams_compare = extract_collocation(ngrams, ngram_finder, args.k)
    
    ngrams_compare.to_excel('./top_collocations_with_punct.xlsx', encoding='utf-8')
    
    if args.conllu is not None:
        assert args.n == 2, "Only Bigram is currently supported!"
        conllulist, sentlist = load_check_data(args.conllu)
        check = pd.DataFrame()
        for columns in list(ngrams_compare.columns):
            if columns is not 'Score':
                check_results = []
                for collocation in ngrams_compare[columns]:
                    check_results.append(check_collocation(collocation, conllulist, sentlist))
                check[columns] = ngrams_compare[columns]
                check[columns+' Result'] = check_results
        check.to_excel('./check_collocations_with_punct.xlsx', encoding='utf-8')
        
                
    


