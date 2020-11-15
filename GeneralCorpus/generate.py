from nltk.corpus import gutenberg
from nltk.corpus import webtext

general_sentences = []
for corpus in [gutenberg, webtext]:
    for fileid in corpus.fileids():
        sentences = corpus.raw(fileid)
        sentences = sentences.split('\n')        
        
        for sentence in sentences:
            general_sentences.append(sentence)        

print(len(general_sentences))
with open('general.txt', 'w') as f:
    for sentence in general_sentences:
        f.write(sentence + "\n")