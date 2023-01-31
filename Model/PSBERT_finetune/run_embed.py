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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
sys.path.append("../PSBERT")

import modeling
import optimization
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import sys
import pickle as pkl
import time
from run_pretrain import *

def del_flags(FLAGS, keys_list):
    for keys in keys_list:
        FLAGS.__delattr__(keys)
    return

def main(_):

    mode = tf.estimator.ModeKeys.EVAL
    input_files = FLAGS.test_input_file + "." + FLAGS.src_bizdate
    features = input_fn(input_files, is_training=False)

    # modeling
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename + "." + FLAGS.src_bizdate
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    # must have checkpoint
    if FLAGS.init_checkpoint==None:
        raise ValueError("Must need a checkpoint for evaluation")

    bert_model, total_loss = model_fn(features, mode, bert_config, vocab, FLAGS.init_checkpoint, FLAGS.learning_rate,
                                      FLAGS.num_train_steps, FLAGS.num_warmup_steps, False, False)

    sequence_output = bert_model.get_sequence_output()
    print(sequence_output)

    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        address_id_list = []
        self_sequence_embedding_list = []
        avg_sequence_embedding_list = []
        avg_del_self_sequence_embedding_list = []

        iter = 0
        start = time.time()
        while True:
            try:

                address_id_v, sequence_output_v, input_ids = sess.run([features["address"], sequence_output, features["input_ids"]])
                seq_length = list(np.sum(input_ids != 0, axis=1))

                address_id_list.append(np.squeeze(address_id_v))

                avg_embedding_list = []
                avg_del_self_embedding_list = []

                for idx in range(len(seq_length)):
                    length = seq_length[idx]
                    avg_embed = np.mean(sequence_output_v[idx, :length,:], axis=0)
                    avg_del_self_embed = np.mean(sequence_output_v[idx, 1:length, :], axis=0)

                    avg_embedding_list.append(avg_embed)
                    avg_del_self_embedding_list.append(avg_del_self_embed)

                self_sequence_embedding_list.append(sequence_output_v[:, 0, :])
                avg_sequence_embedding_list += avg_embedding_list
                avg_del_self_sequence_embedding_list += avg_del_self_embedding_list

                if iter % 500 == 0:
                    end = time.time()
                    print("iter=%d, time=%.2fs" % (iter, end - start))
                    start = time.time()

                iter += 1

            except Exception as e:
                # print(e)
                print("Out of sequence.")
                # save model
                # saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(iter)))
                break

        print("========Embedding Generation Results==========")
        self_sequence_embedding_list = np.concatenate(self_sequence_embedding_list, axis=0)
        avg_sequence_embedding_list = np.array(avg_sequence_embedding_list)
        avg_del_self_sequence_embedding_list = np.array(avg_del_self_sequence_embedding_list)

        address_id_list = np.concatenate(address_id_list, axis=0)
        address_list = vocab.convert_ids_to_tokens(address_id_list)

        print("sample_num=%d" % (self_sequence_embedding_list.shape[0]))
        # write to file

        print("saving embedding and address..")
        ckpt_dir = str(FLAGS.init_checkpoint.split("/")[0])
        model_index = str(FLAGS.init_checkpoint.split("/")[-1].split("_")[1])

        # np.save("./data/bert_embedding_" + model_index + "_" + FLAGS.bizdate + ".npy", self_sequence_embedding_list)
        # np.save("./data/bert_avg_embedding_" + model_index + "_" + FLAGS.bizdate + ".npy", avg_sequence_embedding_list)

        embedding_file_name = "./data/bert_siamese_deself_embedding_" + ckpt_dir + "_" + model_index + "_" + FLAGS.bizdate + ".npy"

        print("saving as: " + embedding_file_name)

        np.save(embedding_file_name, avg_del_self_sequence_embedding_list)
        np.save("./data/address_bert_siamese_" + ckpt_dir + "_" + model_index + "_" + FLAGS.bizdate + ".npy", address_list)
        # np.save("./data/seq_length_" + model_index + "_" + FLAGS.bizdate + "_test.npy", seq_length_list)


    return

if __name__ == '__main__':

    del_flags(FLAGS, ["do_train", "do_eval", "test_input_file", "neg_sample_num", "init_checkpoint", "bert_config_file", "data_dir"])
    flags.DEFINE_bool("do_train", False, "")
    flags.DEFINE_bool("do_eval", True, "")
    flags.DEFINE_string("test_input_file", "../bert/data/eth.embed.tfrecord", "Example for embedding generation.")
    flags.DEFINE_integer("neg_sample_num", 5000, "The number of negative samples in a batch")
    flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
    flags.DEFINE_string("data_dir", '../bert/data/', "data dir.")
    flags.DEFINE_string("src_bizdate", "new", "the sigature for source dataset")
    flags.DEFINE_string(
        "bert_config_file", "../bert/bert_config.json",
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

    tf.app.run()