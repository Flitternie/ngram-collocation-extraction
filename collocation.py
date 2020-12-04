#!/bin/env python

import argparse
import pandas as pd
import nltk
import json
import conllu as cl

def load_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    corpus = []
    for sentence in data:
        corpus.append([(token['form'], token['upos']) for token in sentence])
    unlist_corpus = [token for sent in corpus for token in sent]
    return unlist_corpus

def ngrams(corpus, n):
    assert n==2 or n==3, "Only Bigram and Trigram supported."
    if n==2:
        ngrams = nltk.collocations.BigramAssocMeasures()
        ngram_finder = nltk.collocations.BigramCollocationFinder.from_words(corpus)
    elif n==3:
        ngrams = nltk.collocations.TrigramAssocMeasures() 
        ngram_finder = nltk.collocations.TrigramCollocationFinder.from_words(corpus)
    return ngrams, ngram_finder

def extract_collocation(ngrams, ngram_finder):
    freq = ngram_finder.ngram_fd.items()
    freq_table = pd.DataFrame(list(freq), columns=['ngram','freq']).sort_values(by='freq', ascending=False)
    freq_top = freq_table[:10].ngram.values
    freq_top_score = freq_table[:10]['freq'].values
    
    #PMI
    pmi_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.pmi)), columns=['ngram','PMI']).sort_values(by='PMI', ascending=False)
    pmi_top = pmi_table[:10].ngram.values
    pmi_top_score = pmi_table[:10]['PMI'].values
    
    #t-test
    t_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.student_t)), columns=['ngram','t']).sort_values(by='t', ascending=False)
    t_top = t_table[:10].ngram.values
    t_top_score = t_table[:10]['t'].values
    
    #chi-square
    chi_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.chi_sq)), columns=['ngram','chi-sq']).sort_values(by='chi-sq', ascending=False)
    chi_top = chi_table[:10].ngram.values
    chi_top_score = chi_table[:10]['chi-sq'].values
    
    #likelihood
    lik_table = pd.DataFrame(list(ngram_finder.score_ngrams(ngrams.likelihood_ratio)), columns=['ngram','likelihood']).sort_values(by='likelihood', ascending=False)
    lik_top = lik_table[:10].ngram.values
    lik_top_score = lik_table[:10]['likelihood'].values
    
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
                if (conllu[pair_idx]['form'], conllu[pair_idx]['upos']) == col[abs(word_rank-1)] :
                    if conllu[idx]['deprel'] in result.keys():
                        result[conllu[idx]['deprel']] += 1 
                    else:
                        result[conllu[idx]['deprel']] = 1 
    return result if result != {} else "No Relation Found"
                
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="collocation.py")
    parser.add_argument('-corpus', required=True, type=str,
                        help="path to the corpus")
    parser.add_argument('-conllu', default=None, type=str,
                        help="path to the CoNLL-U file for collocation checking")
    parser.add_argument('-n', required=True, default=2, type=int,
                        help="number of sentences to be generated")
    args = parser.parse_args()

    corpus = load_data(args.corpus)
    ngrams, ngram_finder = ngrams(corpus, args.n)
    ngrams_compare = extract_collocation(ngrams, ngram_finder)
    
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
                
    


