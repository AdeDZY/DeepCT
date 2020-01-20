import argparse
import json
import re
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

stopwords = set([line.strip() for line in open("./data/stopwords.txt")]) 
stemmer = PorterStemmer()

def text_clean(text, stem, stop):
    text = text.replace("\'s", " ")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text) 
    if stop:
       tokens = [t for t in tokens if t.lower() not in stopwords]
    if stem:
        new_tokens = [stemmer.stem(t.lower()) for t in tokens]
    else:
        new_tokens = [t.lower() for t in tokens]
    return ' '.join(new_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("trec_file")
    parser.add_argument("--stem", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()

    # read trec
    
    for line in open(args.trec_file):
        items = line.split('#')
        trec_str = items[0]
        qid = trec_str.split('\t')[0]
        docid = trec_str.split('\t')[2]
        docid2term_recall = {}
   
        json_str = '#'.join(items[1:])
        json_dict = json.loads(json_str)
        qtext = text_clean(json_dict["query"], args.stem, args.stop)
        qtokens = set(qtext.split(' '))
        title_text = json_dict['doc']['title']
        title_text = text_clean(title_text, False, args.stop)
        title_tokens = set(title_text.split(' '))

        for ttoken in title_tokens:
            ttoken2 = ttoken
            if args.stem: ttoken2 = stemmer.stem(ttoken)
            if ttoken2 in qtokens:
                docid2term_recall[ttoken] = docid2term_recall.get(ttoken, 0) + 1
    
        json_dict["term_recall"] = docid2term_recall 
        out_str = json.dumps(json_dict)
        print out_str
