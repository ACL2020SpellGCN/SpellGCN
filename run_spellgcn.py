# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  # You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, time
import modeling
import optimization
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy, time
import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from tensorflow.python.estimator import training
from tensorflow.core.protobuf import rewriter_config_pb2
import sys
reload(sys)
sys.setdefaultencoding("utf8")
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "graph_dir", None,
    "The dir for graph data")

flags.DEFINE_string(
    "eval_file", None,
    "TF example files for eval (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")
flags.DEFINE_string("job_name", None, "The name of the job to train.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "continual", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 128,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")
flags.DEFINE_integer(
    "max_nodes", 4755,
    "Maximum number of adjacent nodes. "
    "Must match data generation.")

flags.DEFINE_integer(
    "max_relations", 2,
    "Maximum number relations.")
flags.DEFINE_integer("next_sent_type", 3, "Next sentence task: support 2 or 3 only. Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the test set.")

flags.DEFINE_bool("with_error", False, "Whether to run eval on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")


flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 10,
                     "Maximum number of checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

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

flags.DEFINE_bool("use_horovod", False, "Whether to use Horovod for multi-gpu training.")

flags.DEFINE_integer("random_seed", None, "Random seed for data generation.")

class LossHook(tf.train.SessionRunHook):
    def __init__(self, intervel):
        self.intervel = intervel
        self.step = 0
        self.fetches = tf.train.SessionRunArgs(
          fetches=[
          "global_step:0",
          "total_loss:0",
          "masked_lm_loss:0",
          "learning_rate:0"
        ])
        self.step_start_time = -1

    def before_run(self, run_context):
      self.step_start_time = time.time()
      return self.fetches

    def after_run(self, run_context, run_values):
        if (not FLAGS.use_horovod or hvd.rank()==0) and self.step % self.intervel == 0:
            global_step, total_loss, masked_lm_loss, learning_rate = run_values.results
            tf.logging.info('global_step=%d\ttotal_loss=%2.6f\tmasked_lm_loss=%2.6f\tlearning_rate=%.6e' % (
                global_step, total_loss, masked_lm_loss, learning_rate))
        self.step += 1


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, masked_lm_positions, masked_lm_ids, masked_lm_weights=None, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_ids = masked_lm_ids
    self.masked_lm_weights = masked_lm_weights
    self.text_b = text_b
    self.label = label

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
               masked_lm_positions,
               masked_lm_ids,
               masked_lm_weights,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_ids = masked_lm_ids
    self.masked_lm_weights = masked_lm_weights
    self.label_id = label_id
    self.is_real_example = is_real_example


# TODO: L2
def gcnLayer(gcn_in, in_dim, gcn_dim, batch_size, max_nodes, 
	max_labels, adj_mat, w_gating=True, num_layers=1, dropout=1.0, name="GCN"):
  out = []
  out.append(gcn_in)
   
  for layer in range(num_layers):
    gcn_in = out[-1]
    if len(out) > 1: in_dim = gcn_dim

    with tf.name_scope("%s-%d" %(name, layer)):
      #with tf.variable_scope("Loop-name-%s-layer-%d" %(name, layer)) as scope:
      #  w_loop = tf.get_variable("w_loop", initializer=tf.eye(in_dim), trainable=False)
      #  inp_loop = tf.tensordot(gcn_in, w_loop, axes=[1,0])
      #  if dropout != 1.0: inp_loop = tf.nn.dropout(inp_loop, keep_probs=dropout)
      #  loop_act = inp_loop
      act_sum = out[-1]
      _act_sum = tf.zeros_like(gcn_in)
      #else:
      #  act_sum = tf.zeros_like(gcn_in)
      #_act_sum = tf.zeros_like(gcn_in)
      for lbl in range(FLAGS.max_relations):
         with tf.variable_scope("label-%d-name-%s_layer-%d" %(lbl, name, layer), reuse=tf.AUTO_REUSE) as scope:
           w_in = tf.get_variable('w_in', initializer=tf.eye(gcn_dim), trainable=True, )
           b_in = tf.get_variable('b_in', [1, gcn_dim], trainable=True, initializer=tf.zeros_initializer())

           if w_gating:
             w_gin = tf.get_variable('w_gin', [gcn_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
             b_gin = tf.get_variable('b_gin', [1], initializer=tf.zeros_initializer())
        
         with tf.name_scope('arcs-%d_name-%s_layer-%d' %(lbl, name, layer)):
           inp_in = tf.matmul(gcn_in, w_in) + b_in # [n, d]
           if adj_mat is not None:
             adj_matrix = adj_mat[lbl]
             in_t = tf.matmul(tf.cast(adj_matrix, tf.float32), inp_in) #[n, d]
           else:
             in_t = inp_in
           
           if dropout != 1.0: in_t = tf.nn.dropout(in_t, keep_prob=dropout)
           if w_gating and adj_mat is not None:
             inp_gin = tf.sigmoid(tf.matmul(gcn_in, w_gin) + tf.to_float(b_gin[None,:])) #[n, 1]
             in_act = tf.matmul(tf.cast(adj_matrix, tf.float32), inp_gin * inp_in) #[n, d]
           else:
             in_act = in_t
         if lbl == 0:
           multi_graph_embs = in_act[None,:,:]
         else:
           multi_graph_embs = tf.concat([multi_graph_embs, in_act[None,:,:]], axis=0) #[MAX_REL, n, d]
      
      with tf.variable_scope("relation_prototype_scope", reuse=tf.AUTO_REUSE):
        w0 = tf.get_variable(
          "relation_prototype",
          shape=[in_dim, 1],
          initializer=tf.contrib.layers.xavier_initializer())
        graph_weights = tf.matmul(multi_graph_embs, w0) / 3
        graph_weights = tf.nn.softmax(graph_weights, axis=0)
        _act_sum = tf.reduce_sum(graph_weights * multi_graph_embs, axis=0) 

      act_sum += _act_sum
      if layer != 0:
        act_sum += out[0]

      #gcn_out = tf.nn.relu(act_sum) if layer != num_layers -1 else act_sum
      gcn_out = act_sum
      out.append(gcn_out)
  return out[-1]
             

def build_gcn_output(adj_mat, w2n, n2w, embedding_table, bert_config, is_training):
    for lbl in range(FLAGS.max_relations):
      adj = adj_mat[lbl] 
      #adj = adj / (adj_mat.sum(0, keepdims=True) + 1e-6)
      adj = adj + np.identity(FLAGS.max_nodes, np.int)
      adj_mat[lbl] = adj

    _adj_mat = tf.to_float(tf.convert_to_tensor(adj_mat)) 
    D = tf.pow(tf.reduce_sum(_adj_mat, axis=2) + 1e-8, -0.5)
    D = tf.matrix_diag(D)
    _adj_mat = tf.keras.backend.batch_dot(_adj_mat, D)
    #print(_adj_mat)
    _adj_mat = tf.transpose(_adj_mat, [0,2,1])
    _adj_mat = tf.keras.backend.batch_dot(_adj_mat,D)

    _w2n = tf.to_int32(tf.convert_to_tensor(w2n))
    _n2w = tf.to_int32(tf.convert_to_tensor(n2w))
    with tf.variable_scope("gcn"):
      node_embedding = gcnLayer(
	  gcn_in = tf.gather(embedding_table, _n2w), #n2w'size is FLAGS.nodes, max value is 21127
	  in_dim=bert_config.hidden_size, 
	  gcn_dim=bert_config.hidden_size, 
          batch_size=32,
          max_nodes=FLAGS.max_nodes,
          max_labels=FLAGS.max_relations,
	  adj_mat=_adj_mat,
          w_gating=False,
          num_layers=2,
          dropout = 0.9 if is_training else 1.0,
          name="GCN") #return [FLAGS.max_nodes, hidden_size]
    
    rest_embedding = embedding_table
    """
    with tf.variable_scope("gcn", reuse=True):
      rest_embedding = gcnLayer(
	  gcn_in = model.get_embedding_table(), #n2w'size is FLAGS.nodes, max value is 21127
	  in_dim=bert_config.hidden_size, 
	  gcn_dim=bert_config.hidden_size, 
          batch_size=32,
          max_nodes=FLAGS.max_nodes,
          max_labels=FLAGS.max_relations,
	  adj_mat=None,
          w_gating=False,
          num_layers=1,
          dropout = 1.0 if is_training else 1.0,
          name="GCN") #return [FLAGS.max_nodes, hidden_size]
    """
    expanded_node_embedding = tf.gather(node_embedding, _w2n) #w2n's is is 21127, max value is FLAGS.nodes
    mask_nodes_ids = tf.to_float(tf.not_equal(_w2n,0))
    gcn_embedding = mask_nodes_ids[:,None] * expanded_node_embedding + (1 - mask_nodes_ids[:,None]) *  rest_embedding

    return gcn_embedding


def model_fn_builder(adj_mat, w2n, n2w, bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    #next_sentence_labels = features["next_sentence_labels"]
   
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    
    gcn_embedding = build_gcn_output(adj_mat, w2n, n2w, model.get_embedding_table(), bert_config, is_training)
 
    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), gcn_embedding,
         masked_lm_positions, masked_lm_ids, masked_lm_weights)


    masked_lm_loss = tf.identity(masked_lm_loss, name="masked_lm_loss")


    total_loss = masked_lm_loss

    total_loss = tf.identity(total_loss, name='total_loss')

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint and (not FLAGS.use_horovod or hvd.rank() == 0):
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if not FLAGS.use_horovod or hvd.rank() == 0:
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
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu, FLAGS.use_horovod)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
      return output_spec
    elif mode == tf.estimator.ModeKeys.PREDICT:

      #def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
      #              masked_lm_weights):#, next_sentence_example_loss,
                    #next_sentence_log_probs, next_sentence_labels):
      """Computes the loss and accuracy of the model."""
      #masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
      #                                   [-1, masked_lm_log_probs.shape[-1]])
      masked_lm_predictions = tf.argmax(
         masked_lm_log_probs, axis=-1, output_type=tf.int32)
        #    values=next_sentence_example_loss)

      predictions = {
          "input_ids": tf.reshape(input_ids, [-1]),
          "predictions": masked_lm_log_probs
      }

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions)
          #eval_metric_ops=eval_metrics)
      return output_spec
    else:
      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
        }

      eval_metrics = metric_fn(
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metric_ops=eval_metrics)

      return output_spec

  return model_fn



def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""
  def get_train_full_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()


  def get_mode(self):
    return "classification"
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


class CSCProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"
    
  def get_dev_full_examples(self, data_dir):
    """See base class."""
    if FLAGS.with_error:
      input_file = os.path.join(data_dir, "TestInputWithError.txt") 
      truth_file = os.path.join(data_dir, "TestTruthWithError.txt") 
    else:
      input_file = os.path.join(data_dir, "TestInput.txt") 
      truth_file = os.path.join(data_dir, "TestTruth.txt") 
     
    #input_file = os.path.join(data_dir, "Training/B1_training_input.txt") 
    #truth_file = os.path.join(data_dir, "Training/B1_training_truth.txt") 
    examples = []
    for i,(inp,trth) in enumerate(
	zip(open(input_file), open(truth_file))):
      inp_fields = inp.strip().split("\t")
      trth_fields = trth.strip().split(" ")
      # Only tackle the sentence withError.
      #if len(trth_fields) == 2:
      #  continue 
      guid = inp_fields[0]  
      text_a = tokenization.convert_to_unicode(inp_fields[1])
      text_a = text_a[:FLAGS.max_seq_length-2]
      masked_lm_positions = [j + 1 for j in range(len(text_a))]
      masked_lm_tokens = [t for t in text_a]
      masked_lm_weights = [1.0 for t in text_a]

      text_b = None
      label = "0"
      for j, pos_tok in enumerate(trth_fields[1:]):
        if j % 2 == 0: 
          pos = pos_tok.strip(",")
        else:
          # replace mispell tokens of the input sentence to correct tokens.
          if int(pos) < FLAGS.max_seq_length and int(pos) <= len(masked_lm_tokens):
            masked_lm_tokens[int(pos)-1] = pos_tok.strip(",").decode("utf8")
      examples.append(
          InputExample(guid=guid, 
			text_a=text_a, 
			masked_lm_positions=masked_lm_positions, 
			masked_lm_ids=masked_lm_tokens, 
			masked_lm_weights=masked_lm_weights,
			text_b=text_b,
			label=label))
    return examples

  def get_train_full_examples(self, data_dir):
    """See base class."""
    if not FLAGS.continual:
      input_file = os.path.join(data_dir, "TrainingInputAll.txt") 
      truth_file = os.path.join(data_dir, "TrainingTruthAll.txt") 
    else:
      input_file = os.path.join(data_dir, "TrainingInputWithError.txt") 
      truth_file = os.path.join(data_dir, "TrainingTruthWithError.txt")
    
    examples = []
    for i,(inp,trth) in enumerate(
	zip(open(input_file), open(truth_file))):
      inp_fields = inp.strip().split("\t")
      trth_fields = trth.strip().split(" ")
     
      guid = inp_fields[0]  
      text_a = tokenization.convert_to_unicode(inp_fields[1])
      text_a = text_a[:FLAGS.max_seq_length-2]
      masked_lm_positions = [j + 1 for j in range(len(text_a))]
      masked_lm_tokens = [t for t in text_a]
      masked_lm_weights = [1.0 for t in text_a]
      text_b = None
      label = "0"
      for j, pos_tok in enumerate(trth_fields[1:]):
        if j % 2 == 0: 
          pos = pos_tok.strip(",")
        else:
        # replace mispell tokens of the input sentence to correct tokens.
          if int(pos) < FLAGS.max_seq_length:
            masked_lm_tokens[int(pos)-1] = pos_tok.strip(",").decode("utf8")
      examples.append(
          InputExample(guid=guid, 
			text_a=text_a, 
			masked_lm_positions=masked_lm_positions, 
			masked_lm_ids=masked_lm_tokens, 
			masked_lm_weights=masked_lm_weights,
			text_b=text_b,
			label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

def convert_single_example(ex_index, example, label_list, output_mode, max_seq_length, max_predictions_per_seq,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  #BERT will add # at head of the number
  #tokens_a = tokenizer.tokenize(example.text_a) 
  tokens_a = [a for a in example.text_a]
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

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
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
  
  
  for j, tok in enumerate(tokens):
    if tok not in tokenizer.vocab:
      tokens[j] = "[UNK]"
       
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)
  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  
  #masked_lm_ids = tokenizer.tokenize("".join(example.masked_lm_ids).decode("utf8"))
  masked_lm_ids = []
  masked_lm_ids.append("[CLS]")
  for tok in example.masked_lm_ids:
     masked_lm_ids.append(tok if tok in tokenizer.vocab else "[UNK]")
  masked_lm_ids.append("[SEP]")
  masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_ids)  
     

  masked_lm_weights = [1.0] * len(masked_lm_ids)
  masked_lm_weights[0] = 0.0
  masked_lm_weights[len(masked_lm_ids)-1] = 0.0
  masked_lm_positions = [i for i in range(len(masked_lm_ids))]

  while len(masked_lm_positions) < max_predictions_per_seq:
    masked_lm_positions.append(0)
    masked_lm_ids.append(0)
    masked_lm_weights.append(0.0)

  assert len(masked_lm_ids) == max_predictions_per_seq
  assert len(masked_lm_positions) == max_predictions_per_seq
  assert len(masked_lm_weights) == max_predictions_per_seq

  if output_mode == 'classification':
    label_id = label_map[example.label]
  elif output_mode == 'regression':
    label_id = float(example.label)

  else:
    raise KeyError(mode)
  if ex_index < 20:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("masked_lm_positions: %s" % " ".join([str(x) for x in masked_lm_positions]))
    tf.logging.info("masked_lm_ids: %s" % " ".join([str(x) for x in masked_lm_ids]))
    tf.logging.info("masked_lm_tokens: %s" % " ".join(
	[str(x) for x in tokenizer.convert_ids_to_tokens(masked_lm_ids)]))
    tf.logging.info("masked_lm_weights: %s" % " ".join([str(x) for x in masked_lm_weights]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      masked_lm_positions=masked_lm_positions,
      masked_lm_ids=masked_lm_ids,
      masked_lm_weights=masked_lm_weights,
      label_id=label_id,
      is_real_example=True)
  return feature



def file_based_convert_examples_to_features(
    examples, label_list, output_mode, max_seq_length, max_predictions_per_seq, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list, output_mode,
                                     max_seq_length, max_predictions_per_seq, tokenizer)

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
    features["masked_lm_positions"] = create_int_feature(feature.masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(feature.masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(feature.masked_lm_weights)
    if output_mode == 'classification':
      features["label_ids"] = create_int_feature([feature.label_id])
    elif output_mode == 'regression':
      features["label_ids"] = create_float_feature([feature.label_id])
    else:
      raise KeyError(mode)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, max_predictions_per_seq, is_training,
                                drop_remainder, output_mode):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "masked_lm_positions": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_ids": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }
  if output_mode == 'classification':
    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.int64)
  elif output_mode == 'regression':
    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.float32)
  else:
    raise KeyError(output_mode)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      print(name)
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    #batch_size = params["batch_size"]
    batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      if FLAGS.use_horovod:
        d = d.shard(hvd.size(), hvd.rank())
      d = d.repeat()
      d = d.shuffle(buffer_size=100, seed=FLAGS.random_seed)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    return d

  return input_fn


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

def set_rand_seed(seed):
  tf.set_random_seed(seed)

def load_graph():
  # Load adjacent matrix
  nodes_vocab = {}
  with open("%s/nodes_vocab.txt"%(FLAGS.graph_dir)) as f:
    for i, line in enumerate(f):
      nodes_vocab.setdefault(line.strip(), i)
  rels_vocab = {}
  with open("%s/relation_vocab.txt"%(FLAGS.graph_dir)) as f:
    for i, line in enumerate(f):
      if len(line.split(" ")) == 1:
        rels_vocab.setdefault(line.strip(), i)
      else:
        rels_vocab.setdefault(line.strip().split(" ")[0], int(line.strip().split(" ")[1]))
  np_adj_mat = np.zeros([len(set(rels_vocab.values())), FLAGS.max_nodes, FLAGS.max_nodes], dtype=np.float)
  with open("%s/spellGraphs.txt"%(FLAGS.graph_dir)) as f:
    for i, line in enumerate(f):
      e1,e2, rel = line.strip().split("|")
      if rel in rels_vocab:
        np_adj_mat[rels_vocab[rel], nodes_vocab[e1], nodes_vocab[e2]] = 1
        np_adj_mat[rels_vocab[rel], nodes_vocab[e2], nodes_vocab[e1]] = 1

  w2n = []
  vocab = {}
  with open("%s/vocab.txt"%(FLAGS.graph_dir)) as f:
    for i, line in enumerate(f):
      word = line.strip()
      vocab.setdefault(word, i)
      if word in nodes_vocab:
        w2n.append(nodes_vocab[word])
      else:
        w2n.append(0)
  n2w = []
  with open("%s/nodes_vocab.txt"%(FLAGS.graph_dir)) as f:
    for i, line in enumerate(f):
      word = line.strip()
      if word in vocab:
        n2w.append(vocab[word])
      else:
        n2w.append(0)
  return np_adj_mat, w2n, n2w

def main(_):
  if FLAGS.random_seed is not None:
    set_rand_seed(FLAGS.random_seed)
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('config: \n' + tf.flags.FLAGS.flags_into_string())
  if FLAGS.use_horovod:
    hvd.init()

  #if not FLAGS.do_train and not FLAGS.do_eval:
  #  raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  processor = CSCProcessor()
    # Build train_input_fn
  if FLAGS.do_train:
    train_examples = processor.get_train_full_examples(FLAGS.data_dir)
    num_train_steps = int(
          len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    num_warmup_steps = int(0.1 * num_train_steps)
  else:
    num_train_steps = 0
    num_warmup_steps = 0

  input_files, eval_files = [], []
  #for input_pattern in FLAGS.input_file.split(","):
  #  input_files.extend(tf.gfile.Glob(input_pattern))

  #for input_pattern in FLAGS.eval_file.split(","):
  #  eval_files.extend(tf.gfile.Glob(input_pattern))

    

  config = tf.ConfigProto()
  if FLAGS.use_horovod:
    config.gpu_options.visible_device_list = str(hvd.local_rank())

  run_config = tf.estimator.RunConfig(
      session_config=config,
      log_step_count_steps=1<<25,
      tf_random_seed=FLAGS.random_seed,
      model_dir=FLAGS.output_dir,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      #save_checkpoints_steps=FLAGS.save_checkpoints_steps if not FLAGS.use_horovod or hvd.rank() == 0 else (1<<25))
      save_checkpoints_steps= num_train_steps / FLAGS.num_train_epochs)
  
  np_adj_mat, w2n, n2w = load_graph()

  model_fn = model_fn_builder(
      #adj_mat = tf.to_int32(tf.convert_to_tensor(np_adj_mat)),
      #w2n = tf.to_int32(tf.convert_to_tensor(w2n)),
      #n2w = tf.to_int32(tf.convert_to_tensor(n2w)),
      adj_mat = np_adj_mat,
      w2n = np.array(w2n),
      n2w = np.array(n2w),
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate if not FLAGS.use_horovod else FLAGS.learning_rate*hvd.size(),
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config)


  if FLAGS.do_train: 
    train_file = os.path.join(FLAGS.output_dir, "train{0}.tf_record".format(FLAGS.max_seq_length))
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    if not os.path.exists(train_file):
      file_based_convert_examples_to_features(
          train_examples, label_list, processor.get_mode(), FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, tokenizer, train_file)
    train_drop_remainder = False
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True,
        drop_remainder=train_drop_remainder,
        output_mode=processor.get_mode())
    # Build evaluation input_fn
    eval_examples = processor.get_dev_full_examples(FLAGS.data_dir)
    eval_file = os.path.join(FLAGS.output_dir, "eval{0}.tf_record".format(FLAGS.max_seq_length))
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    file_based_convert_examples_to_features(
        eval_examples, label_list, processor.get_mode(), FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, tokenizer, eval_file)
    eval_drop_remainder = False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        output_mode=processor.get_mode())



    training_hooks = []
    if FLAGS.use_horovod and hvd.size() > 1:
      training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))

    training_hooks.append(LossHook(100))
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps, hooks=training_hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=FLAGS.max_eval_steps, start_delay_secs=3600, throttle_secs=(1<<25))
    training.train_and_evaluate(estimator, train_spec, eval_spec)

  elif FLAGS.do_predict:
    test_examples = processor.get_dev_full_examples(FLAGS.data_dir)
    test_file = os.path.join(FLAGS.output_dir, "test{0}.tf_record".format(FLAGS.max_seq_length))
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    file_based_convert_examples_to_features(
        test_examples, label_list, processor.get_mode(), FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, tokenizer, test_file)
    test_drop_remainder = False
    test_input_fn = file_based_input_fn_builder(
        input_file=test_file,
        seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        drop_remainder=test_drop_remainder,
        output_mode=processor.get_mode())
    result = estimator.predict(input_fn=test_input_fn)

    output_test_file = os.path.join(FLAGS.output_dir, "test_results.txt")
    input_file = os.path.join(FLAGS.output_dir, "input_file.txt")
    vob = {}
    with open(FLAGS.vocab_file, "r") as f:
      for i, line in enumerate(f):
        vob.setdefault(i, line.strip())
    
    log_prob_rows, log_prob_row = [], []
    with tf.gfile.GFile(output_test_file, "w") as writer:
      with tf.gfile.GFile(input_file, "w") as input_writer:
        tf.logging.info("***** Eval results *****")
        for i, res in enumerate(result):
          if i % (FLAGS.max_seq_length) == 0:
            if i != 0:
              writer.write("\n")
              input_writer.write("\n")
              log_prob_rows.append(log_prob_row)
            log_prob_row = []
            input_writer.write("%s\t" % (test_examples[int(i/FLAGS.max_seq_length)].guid))
          log_prob_row.append(res["predictions"])

          writer.write("%s " % (vob[np.argmax(res["predictions"])]))
          input_writer.write("%s " % (vob[int(res["input_ids"])]))
        # Add the last row predictions.
        log_prob_rows.append(log_prob_row)

    log_prob_rows = np.array(log_prob_rows)
    print(log_prob_rows.shape) #[instance_num, max_seq_len, vob_num]

    # Process the output files to the format given by CSC open-source code. 
    # By comparing between input(withError) and Predictions.
    labels_file = os.path.join(FLAGS.output_dir, "label_results.txt")
    with tf.gfile.GFile(labels_file, "w") as labels_writer:
      with tf.gfile.GFile(input_file, "r") as org_file:
        with tf.gfile.GFile(output_test_file, "r") as test_file:
          for k, (pred, inp) in enumerate(zip(test_file, org_file)):
            pid, atl = inp.strip().split("\t")
            output_list = [pid.strip()]
            pred = pred.split(" ")
            atl = atl.split(" ")
            for i, (pt, at) in enumerate(zip(pred[1:], atl[1:])):
              if at == "[SEP]" or at == '[PAD]':
                break
              # Post preprocess with unsupervised methods, 
	      #because unsup BERT always predict punchuation at 1st pos
              if i == 0:
                if pt == "。" or pt == "，":
                  continue
              if pt.startswith("##"):
                 pt = pt.lstrip("##")   
              if at.startswith("##"):
                 at = at.lstrip("##")   
              if pt != at:
                output_list.append(str(i+1))
                output_list.append(pt)
                
            if len(output_list) == 1:
              output_list.append("0")
            labels_writer.write(", ".join(output_list) + "\n") 

    #Compute F1-score
    detect_TP, detect_FP, detect_FN = 0, 0, 0
    correct_TP, correct_FP, correct_FN = 0, 0, 0
    detect_sent_TP, sent_P, sent_N, correct_sent_TP = 0, 0, 0, 0
    dc_TP, dc_FP, dc_FN = 0, 0, 0
    correction_file = os.path.join(FLAGS.output_dir, "correction_results.txt")
    correction_writer = tf.gfile.GFile(correction_file, "w")
    for idx, (pred, actual) in enumerate(zip(open(labels_file), 
    	open("%s/TestTruthWithError.txt" %FLAGS.data_dir) if FLAGS.with_error else
    	open("%s/TestTruth.txt" %FLAGS.data_dir))):
      pred_tokens = pred.strip().split(" ")
      actual_tokens = actual.strip().split(" ")
      #assert pred_tokens[0] == actual_tokens[0]
      pred_tokens = pred_tokens[1:]
      actual_tokens = actual_tokens[1:]
      detect_actual_tokens = [int(actual_token.strip(",")) \
		for i,actual_token in enumerate(actual_tokens) if i%2 ==0]
      correct_actual_tokens = [actual_token.strip(",") \
		for i,actual_token in enumerate(actual_tokens) if i%2 ==1]
      detect_pred_tokens = [int(pred_token.strip(",")) \
		for i,pred_token in enumerate(pred_tokens) if i%2 ==0]
      _correct_pred_tokens = [pred_token.strip(",") \
		for i,pred_token in enumerate(pred_tokens) if i%2 ==1]

      # Postpreprocess for ACL2019 csc paper which only deal with last detect positions in test data.
      # If we wanna follow the ACL2019 csc paper, we should take the detect_pred_tokens to:

      
      max_detect_pred_tokens = detect_pred_tokens
      
      correct_pred_zip = zip(detect_pred_tokens, _correct_pred_tokens)
      correct_actual_zip = zip(detect_actual_tokens, correct_actual_tokens)
       
      if detect_pred_tokens[0] !=  0:
        sent_P += 1
        if sorted(correct_pred_zip) == sorted(correct_actual_zip):
          correct_sent_TP += 1
      if detect_actual_tokens[0] != 0:
        if sorted(detect_actual_tokens) == sorted(detect_pred_tokens): 
          detect_sent_TP += 1
        sent_N += 1

   
 
      if detect_actual_tokens[0]!=0:
        detect_TP += len(set(max_detect_pred_tokens) & set(detect_actual_tokens)) 
        detect_FN += len(set(detect_actual_tokens) - set(max_detect_pred_tokens)) 
      detect_FP += len(set(max_detect_pred_tokens) - set(detect_actual_tokens)) 
      
      correct_pred_tokens = []
      #Only check the correct postion's tokens
      for dpt, cpt in zip(detect_pred_tokens, _correct_pred_tokens):
        if dpt in detect_actual_tokens:
          correct_pred_tokens.append((dpt,cpt))



              

      correction_list = [actual.split(" ")[0].strip(",")]
      for dat, cpt in correct_pred_tokens:
        correction_list.append(str(dat))
        correction_list.append(cpt) 
      correction_writer.write(" ,".join(correction_list) + "\n")
       	

      correct_TP += len(set(correct_pred_tokens) & set(zip(detect_actual_tokens,correct_actual_tokens))) 
      correct_FP += len(set(correct_pred_tokens) - set(zip(detect_actual_tokens,correct_actual_tokens)))
      correct_FN += len(set(zip(detect_actual_tokens,correct_actual_tokens)) - set(correct_pred_tokens)) 

      # Caluate the correction level which depend on predictive detection of BERT
      dc_pred_tokens = zip(detect_pred_tokens, _correct_pred_tokens)
      dc_actual_tokens = zip(detect_actual_tokens, correct_actual_tokens)
      dc_TP += len(set(dc_pred_tokens) & set(dc_actual_tokens)) 
      dc_FP += len(set(dc_pred_tokens) - set(dc_actual_tokens)) 
      dc_FN += len(set(dc_actual_tokens) - set(dc_pred_tokens)) 
    
    detect_precision = detect_TP * 1.0 / (detect_TP + detect_FP)
    detect_recall = detect_TP * 1.0 / (detect_TP + detect_FN)
    detect_F1 = 2. * detect_precision * detect_recall/ ((detect_precision + detect_recall) + 1e-8)

    correct_precision = correct_TP * 1.0 / (correct_TP + correct_FP)
    correct_recall = correct_TP * 1.0 / (correct_TP + correct_FN)
    correct_F1 = 2. * correct_precision * correct_recall/ ((correct_precision + correct_recall) + 1e-8)

    dc_precision = dc_TP * 1.0 / (dc_TP + dc_FP + 1e-8)
    dc_recall = dc_TP * 1.0 / (dc_TP + dc_FN + 1e-8)
    dc_F1 = 2. * dc_precision * dc_recall/ (dc_precision + dc_recall + 1e-8)
    if FLAGS.with_error:
      #Token-level metrics
      print("detect_precision=%f, detect_recall=%f, detect_Fscore=%f" %(detect_precision, detect_recall, detect_F1))
      print("correct_precision=%f, correct_recall=%f, correct_Fscore=%f" %(correct_precision, correct_recall, correct_F1))  
      print("dc_joint_precision=%f, dc_joint_recall=%f, dc_joint_Fscore=%f" %(dc_precision, dc_recall, dc_F1))
   
    detect_sent_precision = detect_sent_TP * 1.0 / (sent_P)
    detect_sent_recall = detect_sent_TP * 1.0 / (sent_N)
    detect_sent_F1 = 2. * detect_sent_precision * detect_sent_recall/ ((detect_sent_precision + detect_sent_recall) + 1e-8)

    correct_sent_precision = correct_sent_TP * 1.0 / (sent_P)
    correct_sent_recall = correct_sent_TP * 1.0 / (sent_N)
    correct_sent_F1 = 2. * correct_sent_precision * correct_sent_recall/ ((correct_sent_precision + correct_sent_recall) + 1e-8)

    if not FLAGS.with_error:
      #Sentence-level metrics
      print("detect_sent_precision=%f, detect_sent_recall=%f, detect_Fscore=%f" %(detect_sent_precision, detect_sent_recall, detect_sent_F1))
      print("correct_sent_precision=%f, correct_sent_recall=%f, correct_Fscore=%f" %(correct_sent_precision, correct_sent_recall, correct_sent_F1))  
        
         
        
        
	

if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
