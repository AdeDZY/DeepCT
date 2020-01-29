import argparse
import json
import numpy as np


def subword_weight_to_word_weight(subword_weight_str, m, smoothing, keep_all_terms):
    fulltokens = []
    weights = []
    for item in subword_weight_str.split('\t'):
        token, weight = item.split(' ')
        weight = float(weight)
        token = token.strip()
        if token.startswith('##'):
            fulltokens[-1] += token[2:]
        else:
            fulltokens.append(token)
            weights.append(weight)
    assert len(fulltokens) == len(weights)
    fulltokens_filtered, weights_filtered = [], []
    selected_tokens = {}
    for token, w in zip(fulltokens, weights):
        if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
            continue

        if w < 0: w = 0
        if smoothing == "sqrt":
            tf = int(np.round(m * np.sqrt(w)))
        else:
            tf = int(np.round(m * w))
        
        if tf < 1: 
            if not keep_all_terms: continue
            else: tf = 1

        selected_tokens[token] = max(tf, selected_tokens.get(token, 0))

    return selected_tokens
         
def tsv_to_weighted_doc(dataset_file_path,
                        prediction_file_path,
                        output_file_path,
                        m,
                        smoothing='none',
                        keep_all_terms=False,
                        output_format='tsv'):
    """

    :param dataset_file_path: tsv file
    :param prediction_file_path: json/tsv file of predictions
    :return: None
    """
    dataset = []
    predictions = []
    with open(dataset_file_path) as dataset_file, open(prediction_file_path) as prediction_file, open(output_file_path, 'w') as output_file:
        n, e, a = 0, 0, 0
        for l1, l2 in zip(dataset_file, prediction_file):
            n += 1
            if n % 10000 == 0: 
                print("processe {} lines, {} empty, avg len: {}".format(n, e, float(a)/(n - e)))
            did = l1.split('\t')[0]
            selected_tokens = subword_weight_to_word_weight(l2, m, smoothing, keep_all_terms)
            if not selected_tokens: 
                e += 1
                continue 
            sampled_tokens = []
            for t, tf in selected_tokens.items():
                sampled_tokens += [t] * tf
            dtext = ' '.join(sampled_tokens)
            if output_format == "tsv":
                output_file.write(did + '\t' + dtext.strip()  + '\n')
            elif output_format == "json":
                output_file.write(json.dumps({"id": did, "contents": dtext.strip()}))
                output_file.write("\n")
            a += len(sampled_tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', help='Dataset tsv file (collection.tsv.1)')
    parser.add_argument('prediction_file', help='DeepCT prediction file (test_result.tsv)')
    parser.add_argument('output_file', help='Output File')
    parser.add_argument('m', type=int, help='scaling parameter > 0, recommend 100')
    parser.add_argument('--smoothing', type=str, choices=["none", "sqrt"], help="optionally use sqrt to smooth weights. Paper uses none..")
    parser.add_argument('--keep_all_terms', action='store_true', help="do not allow DeepCT to delete terms. Default: false")
    parser.add_argument('--output_format', type=str, choices=["tsv", "json"], default="tsv")
    args = parser.parse_args()

    assert args.m > 0

    tsv_to_weighted_doc(args.dataset_file,
                        args.prediction_file,
                        args.output_file, 
                        args.m,
                        args.smoothing,
                        args.keep_all_terms,
                        args.output_format)

