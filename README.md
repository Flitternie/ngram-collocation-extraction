# Ngram Collocation Extraction
Extracting collocations automatically from a corpus using mutual information/t-test/chi-square/likelihood ratios methods.

## Tutorial
1. Install package dependencies
```
pip install requirements.txt
```

2. Go ahead and run the model!
```
python collocation.py -corpus ./corpus.json -conllu ./ud.conllu -n 2 
```
where `-corpus` represents the path to the corpus, `-n` represents the parameter for tuning n-gram, `-conllu` (not necessary) represents the path to the CoNLL-U file for collocations' depenency relation checking.

## Remarks
- Only bigram is currently supported for CoNLL-U dependency relation checking.
- Feel free to post an issue if you have any questions.

