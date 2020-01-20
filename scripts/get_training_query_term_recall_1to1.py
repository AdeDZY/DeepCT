import argparse
import json
import re
import string
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

stopwords = set([line.strip() for line in open("../data/stopwords.txt")]) 
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
    parser.add_argument("json_in_file", help="Each line: {\"query\": \"what kind of animals are in grasslands\", \"doc\":{\"title\": Tropical grassland animals (which do not all occur in the same area) include giraffes, zebras, buffaloes, ka...}}")
    parser.add_argument("--stem", action="store_true", help="recommend: true")
    parser.add_argument("--stop", action="store_true", help="recommend: true")
    args = parser.parse_args()

    # read trec
    
    for line in open(args.json_in_file):
        term_recall = {}

        json_dict = json.loads(line)
        qtext = text_clean(json_dict["query"], args.stem, args.stop)
        qtokens = set(qtext.split(' '))
        title_text = json_dict['doc']['title']
        title_text = text_clean(title_text, False, args.stop)
        title_tokens = set(title_text.split(' '))

        for ttoken in title_tokens:
            ttoken2 = ttoken
            if args.stem: 
                ttoken2 = stemmer.stem(ttoken)
            if ttoken2 in qtokens:
                term_recall[ttoken] = term_recall.get(ttoken, 0) + 1
    
        json_dict["term_recall"] = term_recall 
        out_str = json.dumps(json_dict)
        print(out_str)
