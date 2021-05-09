# DeepCT and HDCT: Context-Aware Term Importance Estimation For First Stage Retrieval
This repository contains code for two of our papers: 
- arXiv paper "Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval" [arXiv](https://arxiv.org/abs/1910.10687), 2019
- The WebConf2020 paper "Context-Aware Document Term Weighting for Ad-Hoc Search" [pdf](http://www.cs.cmu.edu/~zhuyund/papers/TheWebConf_2020_Dai.pdf), 2020

*Feb 19, 2019*: Checkout our new WebConf2020 paper ["Context-Aware Document Term Weighting for Ad-Hoc Search" ](http://www.cs.cmu.edu/~zhuyund/papers/TheWebConf_2020_Dai.pdf)! It presents HDCT, which extends DeepCT to support long documents and weakly-supervised training! 

*Feb 19, 2019*: Data and instructions for HDCT will come soon.

*May 21, 2020*: Rankings generaed by HDCT for MS-MARCO-Doc: [here](http://boston.lti.cs.cmu.edu/appendices/TheWebConf2020-Zhuyun-Dai/rankings/)

Term frequency is a common method for identifying the importance of a term in a query or document. But it is a weak signal. This work proposes a Deep Contextualized Term Weighting framework that learns to map BERT's contextualized text representations to context-aware term weights for sentences and passages. 

- DeepCT is a framwork for sentence/passage term weighting. When applied to **passages**, DeepCT-Index produces term weights that can be stored in an ordinary inverted index for passage retrieval. When applied to **query** text, DeepCT-Query generates a weighted bag-of-words query that emphasizes essential terms in the query. 
- HDCT extends DeepCT to support long documents. It index **documents** into an ordinary inverted index for retrieval.

<img src="deepct.png" alt="Illustration of DeepCT" width="500"/>


## DATA 1:DeepCT Retrieved Results (Initial Rankings)

We released the top 1000 documents retrieved by DeepCT for the MS MARCO Passage Ranking dev/eval queries. You can use these as your initial document ranking for downstream re-ranking. The ranking files can be downloaded from [here](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/msmarco_rankings/).

## DATA 2: Weighted MS-MARCO Passage Files

If you want to use the DeepCT-Index weighted MS-MARCO passages (e.g., to build index & run experiments), download them here:
[Virtual Appendix/weighted_documents](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/weighted_documents/)

DeepCT generates a floating-point weight for every term: y_{t,d}. To turn it into an integer TF-like weight, we tested 2 different ways:

1. The paper used `TF_{DeepCT}(t, d) = round(y_{t,d} * 100)` (`sample_100_jsonl.zip`)
2. Later I use `TF_{DeepCT}(t, d) = round(sqrt(y_{t,d}) * 100)` (`sqrt_sample_100_jsonl.zip`). sqrt makes small values highe, e.g., sqrt(0.01)=0.1, so more terms will appear in the document. 

Each line in the json file is the text of a weighted passage. We repeat every word TF_{DeepCT} times, so that these json files can be directly feed into [Anserini](https://github.com/castorini/anserini) to build inverted indexes.

`{"id": "2", "contents": "essay essay essay essay essay essay essay essay essay essay essay essay bomb bomb bomb bomb bomb bomb success manmade manmade possible project project project project project project project project project project project project project project project project project project project project project project project project project project project project project project project   atomic atomic atomic atomic atomic atomic making making making ..."}`


In the paper, we fine-tuned the BM25 parameters (k1, b) for all baselines and DeepCT-index methods. We recommend the following parameters:
- Baseline BM25: k1=0.6, b=0.8
- DeepCT: k1=10, b=0.9
- DeepCT-sqrt: k1=18, b=0.7
- Also see detaills in [issue#2](https://github.com/AdeDZY/DeepCT/issues/2).



## DATA 3: Other MS MARCO passage ranking task data for reproducing

To reproduce DeepCT-Index: The corpus, training files, checkpoints,and predictions can be downloaded from the [Virtual Appendix](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/)

1. `data`: MS MARCO passage ranking corpus, and pre-processed training files to train DeepCT. 
2. `output`: the pre-trained DeepCT model (trained in MS MARCO)
3. `predictions`：the DeepCT predicted weights for the entire MS MARCO passage ranking corpus. 


The tokenization will take a long time. Alternatively, you can download the preprocessed binary training/inference files (`output/train.tf_record`, `predictions/collection_pred_1/predict.tf_record`, `predictions/collection_pred_2/predict.tf_record`).  Comment out the `'file_based_convert_examples_to_features()'` function calles in `run_deepct.py` line 1061-1062,1112-1114.

## Run DeepCT 1: Train DeepCT

The source code uses 
- Python 3
- Tensorflow 1.15.0

To use the sample training code, copy and decompress `data` in the [Virtual Appendix](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/) to the.`./data` under this repo. 

```
export BERT_BASE_DIR=/bos/usr0/zhuyund/uncased_L-12_H-768_A-12
export TRAIN_DATA_FILE=./data/marco/myalltrain.relevant.docterm_recall
export OUTPUT_DIR=./output/marco/

python run_deepct.py \
  --task_name=marcodoc \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
  --data_dir=$TRAIN_DATA_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --recall_field=title \
  --output_dir=$OUTPUT_DIR
```

`BERT_BASE_DIR`: The small, uncased BERT model released by Google.

`TRAIN_DATA_FILE`: a json file with the query term recall labels. It tooks like:

`{"query": "what kind of animals are in grasslands", "term_recall": {"grassland": 1, "animals": 1}, "doc": {"position": "1", "id": "4083643", "title": "Tropical grassland animals (which do not all occur in the same area) include giraffes, zebras, buffaloes, kangaroos, mice, moles, gophers, ground squirrels, snakes, worms, termites, beetles, lions, leopards, hyenas, and elephants."}}`

`term_recall` is the ground-truth term weight (see details in our paper). You can replace `term_recall` with any other ground-truth term weight labels.

`OUTPUT_DIR`: output folder for training. It will store the tokenized training file (train.tf_record) and the checkpoints (model.ckpt).

## Run DeepCT 2: Use DeepCT to Predict Term Weights (Inference)
 
 To use the sample training code, download and decompress `data` in the [Virtual Appendix](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/) to the.`./data` under this repo. Download the pre-trained DeepCT model (model.ckpt-65816 files) from `outputs` to `./output`.
 
(You can skip this step. Alternatively, direct download our DeepCT predicted weights for the entire MS MARCO passage ranking corpus, from `prediction` in the [Virtual Appendix](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/).)
 
 ```
export BERT_BASE_DIR=/bos/usr0/zhuyund/uncased_L-12_H-768_A-12
export INIT_CKPT=./output/marco/model.ckpt-65816
export TEST_DATA_FILE=./data/collection.tsv.1
export OUTPUT_DIR=./predictions/marco/collection_pred_1/

python run_deepct.py \
  --task_name=marcotsvdoc \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$TEST_DATA_FILE \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT_CKPT \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR
 ```
 
 `TEST_DATA_FILE`: a tsv file (docid \t doc_content) of the entire collection that you want ot reweight and index. It looks like:
 
 `0 \t The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.`
 
 `$OUTPUT_DIR`: output folder for testing. It will store the tokenized test file (predict.tf_record) and the predictions (test_results.tsv). test_results.tsv file looks like:
 
 `[CLS] 0.0       the -0.0023216241970658302      presence 0.0160924531519413     of 0.0003044793847948313      ...     ; -0.0012609917903319001        hundreds 0.000732177053578198   of -0.0018553155241534114       thousands 0.001125242910347879  of -0.0011851346353068948       innocent 0.004741794429719448   lives 0.015339942649006844      ob 0.006402325350791216 ##lite 0.0      ##rated 0.0     . -0.002221715170890093 [SEP] -0.0      [PAD] -0.0      [PAD] -0.0      [PAD] -0.0      [PAD] -0.0`
 
 ## Run DeepCT 3: Turn float-point term wegihts into TF-like index weights

Now we need to turn the above BERT outputs (`y`) into TF-like term index weights. 

This methods implements (eq 4) in the paper: `TF_{DeepCT}(t,d) = round(y * N)`. Let us know if you found better ways!

Download our DeepCT predicted weights for the entire MS MARCO passage ranking corpus, from `prediction` in the [Virtual Appendix](http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/
 
 Use `bert_term_sample_to_json.py` to: 
 1. map test_result.tsv back to original document ids, and
 2. scale the weights into integers for indexing.
 
 ```
 usage: bert_term_sample_to_json.py [-h] [--output_format {tsv,json}]
                                   dataset_file prediction_file output_file m

positional arguments:
  dataset_file          Dataset tsv file (collection.tsv.1)
  prediction_file       DeepCT prediction file (test_result.tsv)
  output_file           Output File
  m                     scaling parameter > 0, recommend 100

optional arguments:
  -h, --help            show this help message and exit
  --output_format {tsv,json}
 ```
 
 The output files can be feed to indexing tools such as Anserini (used in paper), Indri, or Lucene to build index and run retrieval.
 Go to the original repository of [Anserini](https://github.com/castorini/anserini)
 
 For retrieval, it is critical to fine-tune the BM25/LM parameters (k1, b, \mu). See detaills in [issue#2](https://github.com/AdeDZY/DeepCT/issues/2).
 
