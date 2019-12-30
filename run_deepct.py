# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import numpy as np
import modeling
import optimization
import tokenization
import tensorflow as tf
import random
import json
import re

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "use_all_layers", False,
    "feature is last layer or concat all encoder layers")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("max_body_length", 500, "cut body at this length")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 20000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "fold", 3,
    "run fold")

flags.DEFINE_string(
    "recall_field", None,
    "title, body, all")

flags.DEFINE_string(
    "doc_field", None,
    "title, body, url")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, term_recall_dict=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a_list: list of strings.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.term_recall_dict = term_recall_dict


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 target_weights,
                 target_mask,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.target_weights = target_weights
        self.target_mask = target_mask
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class CarDocProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        examples = []
        train_files = ["train.fold0.docterm_recall", "train.fold1.docterm_recall", "train.fold2.docterm_recall", "train.fold3.docterm_recall"]

        for file_name in train_files:
            train_file = open(os.path.join(data_dir, file_name))
            for i, line in enumerate(train_file):
                json_dict = json.loads(line)
                docid = json_dict["doc"]["id"]
                doc_text = tokenization.convert_to_unicode(json_dict["doc"]["title"])
                term_recall_dict = json_dict["term_recall"]
                if not term_recall_dict or not doc_text.strip():
                    continue

                guid = "train-%s" % docid
                examples.append(
                    InputExample(guid=guid, text=doc_text, term_recall_dict=term_recall_dict)
                )
            train_file.close()
        random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir):
        dev_files = ["train.fold4.docterm_recall.firsthalf"]
        examples = []

        for file_name in dev_files:
            dev_file = open(os.path.join(data_dir, file_name))
            for i, line in enumerate(dev_file):
                json_dict = json.loads(line)
                docid = json_dict["doc"]["id"]
                doc_text = tokenization.convert_to_unicode(json_dict["doc"]["title"])
                term_recall_dict = json_dict["term_recall"]

                guid = "dev-%s" % docid
                examples.append(
                    InputExample(guid=guid, text=doc_text, term_recall_dict=term_recall_dict)
                )
            dev_file.close()
        return examples

    def get_test_examples(self, data_dir):
        test_files = ["train.fold4.docterm_recall.secondhalf"]
        examples = []

        for file_name in test_files:
            test_file = open(os.path.join(data_dir, file_name))
            for i, line in enumerate(test_file):
                json_dict = json.loads(line)
                docid = json_dict["doc"]["id"]
                doc_text = tokenization.convert_to_unicode(json_dict["doc"]["title"])
                term_recall_dict = json_dict["term_recall"]

                guid = "test-%s" % docid
                examples.append(
                    InputExample(guid=guid, text=doc_text, term_recall_dict=term_recall_dict)
                )
            test_file.close()
        return examples

class MarcoDocProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        examples = []
        train_files = [data_dir]

        for file_name in train_files:
            train_file = open(file_name)
            for i, line in enumerate(train_file):
                json_dict = json.loads(line)
                docid = json_dict["doc"]["id"]
                doc_text = tokenization.convert_to_unicode(json_dict["doc"]["title"])
                term_recall_dict = json_dict["term_recall"]

                guid = "train-%s" % docid
                examples.append(
                    InputExample(guid=guid, text=doc_text, term_recall_dict=term_recall_dict)
                )
            train_file.close()
        random.shuffle(examples)
        return examples


class MarcoTsvDocProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        test_files = [data_dir]
        examples = []

        for file_name in test_files:
            test_file = open(file_name)
            for i, line in enumerate(test_file):
                docid, t = line.strip().split('\t')
                doc_text = tokenization.convert_to_unicode(t)
                term_recall_dict = {}

                guid = "test-%s" % docid
                examples.append(
                    InputExample(guid=guid, text=doc_text, term_recall_dict=term_recall_dict)
                )
            test_file.close()
        return examples

class IdContentsJsonDocProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        test_files = [data_dir]
        examples = []

        for file_name in test_files:
            test_file = open(file_name)
            for i, line in enumerate(test_file):
                jdict = json.loads(line)
                docid = jdict["id"]
                doc_text = jdict[FLAGS.doc_field]
                doc_text = tokenization.convert_to_unicode(doc_text)
                doc_text = truncate_and_clean_trec_19_doc(doc_text, FLAGS.max_body_length)
                term_recall_dict = {}
                if not doc_text.strip():
                     doc_text = '.'
                #    tf.logging.info("skipping {}".format(docid))
                #    continue

                guid = "test-%s" % docid
                examples.append(
                    InputExample(guid=guid, text=doc_text, term_recall_dict=term_recall_dict)
                )
            test_file.close()
        return examples

class CarJsonDocProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        test_files = [data_dir]
        examples = []

        for file_name in test_files:
            test_file = open(file_name)
            for i, line in enumerate(test_file):
                jdict = json.loads(line)
                docid = jdict["id"]
                doc_text = jdict["content"]
                doc_text = tokenization.convert_to_unicode(doc_text)
                term_recall_dict = {}
                if not doc_text.strip():
                     doc_text = '.'

                guid = "test-%s" % docid
                examples.append(
                    InputExample(guid=guid, text=doc_text, term_recall_dict=term_recall_dict)
                )
            test_file.close()
        return examples


class MarcoQueryProcessor(DataProcessor):

    def __init__(self):
        self.recall_field = FLAGS.recall_field
        tf.logging.info("Using recall fields {}".format(self.recall_field))

    def get_train_examples(self, data_dir):
        examples = []
        train_files = ["mytrain.term_recall.json"] 

        for file_name in train_files:
            train_file = open(os.path.join(data_dir, file_name))
            for i, line in enumerate(train_file):
                q_json_dict = json.loads(line)
                qid = q_json_dict["qid"]
                q_text = tokenization.convert_to_unicode(q_json_dict["query"])
                term_recall_dict = q_json_dict["term_recall"][self.recall_field]

                guid = "train-%s" % qid
                examples.append(
                    InputExample(guid=guid, text=q_text, term_recall_dict=term_recall_dict)
                )
            train_file.close()
        random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir):
        dev_files = ["mydev.term_recall.json"] 
        examples = []

        for file_name in dev_files:
            dev_file = open(os.path.join(data_dir, file_name))
            for i, line in enumerate(dev_file):
                q_json_dict = json.loads(line)
                qid = q_json_dict["qid"]
                q_text = tokenization.convert_to_unicode(q_json_dict["query"])
                term_recall_dict = q_json_dict["term_recall"][self.recall_field]

                guid = "dev-%s" % qid
                examples.append(
                    InputExample(guid=guid, text=q_text, term_recall_dict=term_recall_dict)
                )
            dev_file.close()
        return examples

    def get_test_examples(self, data_dir):
        test_files = ["mytest.term_recall.json"] 
        examples = []

        for file_name in test_files:
            test_file = open(os.path.join(data_dir, file_name))
            for i, line in enumerate(test_file):
                q_json_dict = json.loads(line)
                qid = q_json_dict["qid"]
                q_text = tokenization.convert_to_unicode(q_json_dict["query"])
                term_recall_dict = q_json_dict["term_recall"][self.recall_field]

                guid = "test-%s" % qid
                examples.append(
                    InputExample(guid=guid, text=q_text, term_recall_dict=term_recall_dict)
                )
            test_file.close()
        return examples

class QueryProcessor(DataProcessor):

    def __init__(self):
        self.n_folds = 5
        self.fold = FLAGS.fold
        self.recall_fields = FLAGS.recall_field.split(',')
        tf.logging.info("Using recall fields {}".format(self.recall_fields))

        self.train_folds = [(self.fold + i) % self.n_folds + 1 for i in range(self.n_folds - 2)]
        self.dev_fold = (self.fold + self.n_folds - 2) % self.n_folds + 1
        self.test_fold = (self.fold + self.n_folds - 1) % self.n_folds + 1
        tf.logging.info("Train Folds: {}".format(str(self.train_folds)))
        tf.logging.info("Dev Fold: {}".format(str(self.dev_fold)))
        tf.logging.info("Test Fold: {}".format(str(self.test_fold)))

    def get_train_examples(self, data_dir):
        examples = []
        train_files = ["{}.json".format(i) for i in self.train_folds] #+ ["aux.json"]
        data_dirs = data_dir.split(',')

        for file_name in train_files:
            for data_dir in data_dirs:
                train_file = open(os.path.join(data_dir, file_name))
                for i, line in enumerate(train_file):
                    q_json_dict = json.loads(line)
                    qid = q_json_dict["qid"]
                    q_text = tokenization.convert_to_unicode(q_json_dict["query"])
                    for field in self.recall_fields:
                        if field not in q_json_dict["term_recall"]: continue
                        term_recall_dict = q_json_dict["term_recall"][field]

                        guid = "train-%s" % qid
                        examples.append(
                            InputExample(guid=guid, text=q_text, term_recall_dict=term_recall_dict)
                        )
                train_file.close()
        random.shuffle(examples)
        return examples

    def get_dev_examples(self, data_dir):
        dev_files = ["{}.json".format(self.dev_fold)]
        examples = []
        data_dirs = data_dir.split(',')

        for file_name in dev_files:
            for data_dir in data_dirs:
                dev_file = open(os.path.join(data_dir, file_name))
                for i, line in enumerate(dev_file):
                    q_json_dict = json.loads(line)
                    qid = q_json_dict["qid"]
                    q_text = tokenization.convert_to_unicode(q_json_dict["query"])
                    for field in self.recall_fields:
                        if field not in q_json_dict["term_recall"]: continue
                        term_recall_dict = q_json_dict["term_recall"][field]

                        guid = "dev-%s" % qid
                        examples.append(
                            InputExample(guid=guid, text=q_text, term_recall_dict=term_recall_dict)
                        )
                dev_file.close()
        return examples

    def get_test_examples(self, data_dir):
        examples = []
        test_files = ["{}.json".format(self.test_fold)]
        examples = []
        data_dir = data_dir.split(',')[0]

        for file_name in test_files:
            test_file = open(os.path.join(data_dir, file_name))
            for i, line in enumerate(test_file):
                q_json_dict = json.loads(line)
                qid = q_json_dict["qid"]
                q_text = tokenization.convert_to_unicode(q_json_dict["query"])
                for field in self.recall_fields:
                    if field not in q_json_dict["term_recall"]: continue
                    term_recall_dict = q_json_dict["term_recall"][field]

                    guid = "test-%s" % qid
                    examples.append(
                        InputExample(guid=guid, text=q_text, term_recall_dict=term_recall_dict)
                    )
                    break
            test_file.close()
        return examples


#stopwords_path = "/bos/usr0/zhuyund/query_reweight/stopwords2"
#stopwords = [l.strip() for l in open(stopwords_path)]


def gen_target_token_weights(tokens, term_recall_dict):
    fulltoken = tokens[0]
    i = 1
    s = 0
    term_recall_weights = [0 for _ in range(len(tokens))]
    term_recall_mask = [0 for _ in range(len(tokens))]
    while i < len(tokens):
        if tokens[i].startswith('##'):
            fulltoken += tokens[i][2:]
            i += 1
            continue

        w = term_recall_dict.get(fulltoken, 0.0)
        term_recall_weights[s] = w
        #if fulltoken in stopwords:
        #    term_recall_mask[s] = 0
        #else:
        term_recall_mask[s] = 1
        fulltoken = tokens[i]
        s = i
        i += 1

    w = term_recall_dict.get(fulltoken, 0)
    term_recall_weights[s] = w
    term_recall_mask[s] = 1
    return term_recall_weights, term_recall_mask


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            target_weights=[0.0] * max_seq_length,
            target_mask=[0] * max_seq_length,
            is_real_example=False)

    text_tokens = tokenizer.tokenize(example.text)
    if len(text_tokens) == 0:
        text_tokens = ["."]
    if len(text_tokens) > max_seq_length - 2:
        text_tokens = text_tokens[0:(max_seq_length - 2)]
    text_target_weights, text_target_mask = gen_target_token_weights(text_tokens, example.term_recall_dict)
    assert len(text_target_mask) == len(text_tokens)
    assert len(text_target_weights) == len(text_tokens)


    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    target_weights, target_mask = [], []
    tokens.append("[CLS]")
    segment_ids.append(0)
    target_weights.append(0)
    target_mask.append(0)
    for i, token in enumerate(text_tokens):
        tokens.append(token)
        target_weights.append(text_target_weights[i])
        target_mask.append(text_target_mask[i])
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    target_weights.append(0)
    target_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        target_weights.append(0)
        target_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(target_weights) == max_seq_length
    assert len(target_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("target_weights: %s" % " ".join([str(x) for x in target_weights]))
        tf.logging.info("target_mask: %s" % " ".join([str(x) for x in target_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        target_weights=target_weights,
        target_mask=target_mask,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["target_weights"] = create_float_feature(feature.target_weights)
        features["target_mask"] = create_int_feature(feature.target_mask)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_files, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "target_weights": tf.FixedLenFeature([seq_length], tf.float32),
        "target_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    if isinstance(input_files, str):
        input_files = input_files.split(',') 

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.

        if is_training:
            num_cpu_threads = 4
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))
            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=True,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)

        else:
            d = tf.data.TFRecordDataset(input_files)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 target_weights, target_mask, use_one_hot_embeddings, use_all_layers):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    #bert_output_cls = tf.expand_dims(model.get_pooled_output(), 1)
    #tf.logging.info(bert_output_cls.shape)
    if use_all_layers:
        all_encoder_layers = model.get_all_encoder_layers() # [natcj_size, seq_lenght, hidden_size, nlayers]
        bert_output_layer = tf.concat(all_encoder_layers, -1) 
        tf.logging.info(bert_output_layer.shape)
    else:
        bert_output_layer = model.get_sequence_output()  # [batch_size, seq_length, hidden_size]
    #bert_output_layer = bert_output_layer - bert_output_cls
    tf.logging.info(bert_output_layer.shape)
    hidden_size = bert_output_layer.shape[-1].value
    seq_length = bert_output_layer.shape[-2].value
    tf.logging.info(hidden_size)
    tf.logging.info(seq_length)
    bert_output_layer = tf.reshape(bert_output_layer, [-1, hidden_size])  # [batch_size * seq_length, hidden_size]
    tf.logging.info(bert_output_layer.shape)

    output_weights = tf.get_variable(
        "tw_output_weights", [1, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "tw_output_bias", [1], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            bert_output_layer = tf.nn.dropout(bert_output_layer, keep_prob=0.9)

        logits = tf.matmul(bert_output_layer, output_weights, transpose_b=True)  # [batch_size * seq_length, 1]
        logits = tf.nn.bias_add(logits, output_bias)  # [batch_size * seq_length, 1]
        logits = tf.reshape(logits, [-1, seq_length])
        loss = tf.losses.mean_squared_error(
            labels=target_weights,
            predictions=logits,
            weights=target_mask,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        per_example_loss = tf.losses.mean_squared_error(
            labels=target_weights,
            predictions=logits,
            weights=target_mask,
            reduction=tf.losses.Reduction.NONE)
        per_example_loss = tf.reduce_sum(per_example_loss, axis=-1)
        masked_logits = logits * tf.to_float(target_mask)

        probabilities = tf.nn.sigmoid(logits) # [batch_size * seq_length, hidden_size]
        masked_probabilities = probabilities * tf.to_float(target_mask)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        #per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        #loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, masked_logits


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, use_all_layers):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        target_weights = features["target_weights"]
        target_mask = features["target_mask"]

        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, target_weights, target_mask, use_one_hot_embeddings, use_all_layers)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss):
                loss = tf.metrics.mean(values=per_example_loss)
                return {
                    "eval_loss": loss
                }

            eval_metrics = (metric_fn, [total_loss])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"logits": logits, "target_weights": target_weights, "token_ids": input_ids},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_target_weights = []
    all_target_mask = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_target_weights.append(features.target_weights)
        all_target_mask.append(features.target_mask)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "target_weights":
                tf.constant(
                    all_target_weights,
                    shape=[num_examples, seq_length],
                    dtype=tf.float32),
            "target_mask":
            tf.constant(
                all_target_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32,
            )
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {"query": QueryProcessor,
                  "marcoquery": MarcoQueryProcessor,
                  "marcodoc": MarcoDocProcessor, 
                  "marcotsvdoc": MarcoTsvDocProcessor,
                  "cardoc": CarDocProcessor, 
                  "carjsondoc": CarJsonDocProcessor,
                  "idcontentsjson": IdContentsJsonDocProcessor
                  }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()




    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, FLAGS.max_seq_length, tokenizer, train_file)
        #tf.logging.info("write to {}/train.tf_record! exit. I am NOT training".format(FLAGS.output_dir))
        #tf.logging.info("I am NOT writing train file")
        #exit(-1)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu, 
        use_all_layers=FLAGS.use_all_layers)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_files=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples,
                                              FLAGS.max_seq_length, tokenizer,
                                              predict_file)
        #tf.logging.info("I am NOT writing predict.tf_record")
        #tf.logging.info("I am NOT runnign model")
        #exit(-1)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_files=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                targets = prediction["target_weights"]
                logits = prediction["logits"]
                tokens = tokenizer.convert_ids_to_tokens(prediction["token_ids"])
                if i >= num_actual_predict_examples:
                    break
                output_line = '\t'.join(['{0} {1:.5f}'.format(t, w) for (t, w) in zip(tokens, logits) if t != '[PAD]' and t != '[CLS]'])
                writer.write(output_line)
                writer.write('\n')
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
