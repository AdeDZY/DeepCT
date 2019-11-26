# DeepCT: Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval
Repository for our arXiv paper "Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval"

[Find the paper on arXiv](https://arxiv.org/abs/1910.10687)


Abstract: Term frequency is a common method for identifying the importance of a term in a query or document. But it is a weak signal, especially when the frequency distribution is flat, such as in long queries or short documents where the text is of sentence/passage-length. This paper proposes a Deep Contextualized Term Weighting framework that learns to map BERT's contextualized text representations to context-aware term weights for sentences and passages.

When applied to passages, DeepCT-Index produces term weights that can be stored in an ordinary inverted index for passage retrieval. When applied to query text, DeepCT-Query generates a weighted bag-of-words query. Both types of term weight can be used directly by typical first-stage retrieval algorithms. This is novel because most deep neural network based ranking models have higher computational costs, and thus are restricted to later-stage rankers.

Experiments on four datasets demonstrate that DeepCT's deep contextualized text understanding greatly improves the accuracy of first-stage retrieval algorithms.

November 26, 2019: I am still half-way of cleaning up my experimental codes. But, I have received several requests for the code, so here I decided to put what I already have. 
In this version, I provide code, data and instructions for the document reweighting (DeepCT-Index) part, focusing on the MSMARCO passage ranking dataset. -- 
Zhuyun

## MSMARCO passage ranking data
The training files (query term recall labels), checkpoints,and predictions can be downloaded from the [Virtual Appendix]()

## Train DeepCT
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

You can replace `term_recall` with any other ground-truth term weight labels.

`OUTPUT_DIR`: output folder for training. It will store the tokenized training file (train.tf_record) and the checkpoints (model.ckpt).

## Use DeepCT to Predict Term Weights
 
 ```
export BERT_BASE_DIR=/bos/usr0/zhuyund/uncased_L-12_H-768_A-12
export INIT_CKPT=./output/marco/model.ckpt-65816
export TEST_DATA_FILE=./data/collection.tsv.1
export OUTPUT_DIR=./output/marco/collection_pred_1/

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
 
 ## Post-Processing
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