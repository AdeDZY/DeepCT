import argparse
import json
import numpy as np


def subword_weight_to_word_weight(subword_weight_str, m):
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
        if token == '[CLS]' or token == '[SEP]':
            continue
        if w < 0: continue
        if token in [".", "-", "!", ",", ":", ";", "?", "#", "@", "%", "(", ")", "[", "]"]: continue
        tf = int(np.round(m * np.sqrt(w)))
        if tf < 1: continue
        selected_tokens[token] = max(tf, selected_tokens.get(token, 0))

    return selected_tokens
        
def position_decay_avg(tf_arr_pos, total_passages):
    decayed_tf_arr = []
    norm = sum([1.0/float(i) for i in range(1, total_passages + 1)])
    for tf, pos in tf_arr_pos:
        decayed_tf_arr.append(float(tf)/pos) 
    agg_tf = sum(decayed_tf_arr)/norm
    return agg_tf 

def position_decay_sum(tf_arr_pos, total_passages):
    decayed_tf_arr = []
    for tf, pos in tf_arr_pos:
        decayed_tf_arr.append(float(tf)/pos) 
    agg_tf = sum(decayed_tf_arr)
    return agg_tf 

def flat_avg(tf_arr_pos, total_passages):
    agg_tf = sum([tf for tf, _ in tf_arr_pos])/float(total_passages)
    return agg_tf 

def flat_sum(tf_arr_pos, total_passages):
    agg_tf = sum([tf for tf, _ in tf_arr_pos])
    return agg_tf 

def flat_max(tf_arr_pos, total_passages):
    agg_tf = max([tf for tf, _ in tf_arr_pos])
    return agg_tf 


def json_to_trec(dataset_file_path,
                 prediction_file_path,
                 output_file_path,
                 m,
                 output_format,
                 tf_agg_method,
                 max_number_passages):
    """

    :param dataset_file_path: json file
    :param prediction_file_path: json file of predictions
    :return: None
    """
    dataset = []
    predictions = []
    agg_methods = {"avg": flat_avg, "sum":flat_sum, "position_decay_avg": position_decay_avg, "position_decay_sum": position_decay_sum, "max": flat_max}
    with open(dataset_file_path) as dataset_file, open(prediction_file_path) as prediction_file, open(output_file_path, 'w') as output_file:
        n, u, a = 0, 0, 0
        current_docid, used_passages, doc_selected_tokens = "",  0, {}
        curr_title, curr_url = "", ""
        for l1, l2 in zip(dataset_file, prediction_file):
            orj_json = json.loads(l1)
            pid, url, title = orj_json['id'], orj_json.get('url', ''), orj_json.get('title', '')
            docid = pid.split("_")[0]
      
            if docid != current_docid:
                sampled_tokens = []
                for t, arr_tf_pos in doc_selected_tokens.items():
                     tf = agg_methods[tf_agg_method](arr_tf_pos, used_passages)    
                     tf = int(tf)
                     if tf > 0:
                         sampled_tokens += [t] * tf
                         u += 1
                dtext = ' '.join(sampled_tokens)
                if output_format == "tsv" and dtext.strip():
                    output_file.write(current_docid + '\t' + dtext.strip()  + '\n')
                elif output_format == "json" and dtext.strip():
                    output_file.write(json.dumps({"id": current_docid, 'contents': dtext.strip(), 'title': curr_title, 'url': curr_url}))
                    output_file.write("\n")

                a += len(sampled_tokens)
                n += 1
                if n % 1000 == 0: 
                    print("processe {} docs, avg len: {}, avg unique terms: {}".format(n, float(a)/n, float(u)/n))

                doc_selected_tokens = {}
                current_docid, curr_title, curr_url = docid, title, url
                used_passages = 0 

            if used_passages > max_number_passages: 
                continue 

            passage_selected_tokens = subword_weight_to_word_weight(l2, m)
            used_passages += 1
            for t, tf in passage_selected_tokens.items():
                if t not in doc_selected_tokens:
                    doc_selected_tokens[t] = []
                doc_selected_tokens[t].append((tf, used_passages))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', help='Dataset json file')
    parser.add_argument('prediction_file', help='Prediction json File')
    parser.add_argument('output_file', help='Output File')
    parser.add_argument('m', type=int, help='Thred > 0')
    parser.add_argument('--output_format', type=str, choices=["tsv", "json"], default="tsv")
    parser.add_argument('--tf_agg_method', type=str, choices=["avg", "sum", "position_decay_avg", "position_decay_sum", "max"], default="avg")
    parser.add_argument('--max_number_passages', type=int, default=10000)
    args = parser.parse_args()

    assert args.m > 0
    assert args.max_number_passages > 0

    json_to_trec(args.dataset_file,
                 args.prediction_file,
                 args.output_file, 
                 args.m,
                 args.output_format,
                 args.tf_agg_method,
                 args.max_number_passages)

