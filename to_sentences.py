import json
import sys
#import spacy
import re
import nltk

#nlp = spacy.load('en') 

fin = open(sys.argv[1])
fout = open(sys.argv[2], 'w')

for line in fin:
    json_dict = json.loads(line)
    #title = json_dict.get('title', "")
    body = json_dict.get('paperAbstract', "")
    #doc = title + '\n' + body
    #doc = ' '.join(body.split()[0:100])
    #sentences = [sent.string.strip() for sent in doc.sents]
    #fout.write(title)
    #fout.write("\n")
    sentences = nltk.sent_tokenize(body) 
    for sent in sentences:
        fout.write(sent)
        fout.write("\n")
    fout.write("\n")
    
