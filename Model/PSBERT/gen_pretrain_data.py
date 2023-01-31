import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import collections
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import six
import multiprocessing
import time
from vocab import FreqVocab

tf.logging.set_verbosity(tf.logging.INFO)

random_seed = 12345
rng = random.Random(random_seed)

short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("pool_size", 10, "multiprocesses pool size.")
flags.DEFINE_integer("max_seq_length", 50, "max sequence length.")
flags.DEFINE_integer("max_predictions_per_seq", 40, "max_predictions_per_seq.")
flags.DEFINE_float("masked_lm_prob", 0.8, "Masked LM probability.")
flags.DEFINE_float("mask_prob", 1.0, "mask probabaility")
flags.DEFINE_bool("do_eval", True, "")
flags.DEFINE_bool("do_embed", True, "")
flags.DEFINE_integer("dupe_factor", 5, "Number of times to duplicate the input data (with different masks).")
flags.DEFINE_integer("sliding_step", 30, "sliding window step size.")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("dataset_name", 'eth', "dataset name.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_string("bizdate", None, "the date of running experiments")

if FLAGS.bizdate is None:
    raise ValueError("bizdate is required..")

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(
    ",")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address, tokens, masked_lm_positions, masked_lm_labels):

        self.address = [address]
        self.tokens = list(map(lambda x: x[0], tokens))
        self.block_timestamps = list(map(lambda x: x[2], tokens))
        self.values = list(map(lambda x: x[3], tokens))

        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        self.io_flags = list(map(map_io_flag, tokens))
        self.cnts = list(map(lambda x: x[5], tokens))
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = "address: %s\n" % (self.address[0])
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature

def gen_samples(sequences,
                dupe_factor,
                masked_lm_prob,
                max_predictions_per_seq,
                pool_size,
                rng,
                force_head=False):
    instances = []
    # create train
    if force_head:
        for step in range(dupe_factor):
            start = time.time()
            for tokens in sequences:
                (address, tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions_force_head(tokens)
                instance = TrainingInstance(
                    address=address,
                    tokens=tokens,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            end = time.time()
            cost = end - start
            print("step=%d, time=%.2f" % (step, cost))
        print("=======Finish========")

    else:
        for step in range(dupe_factor):
            start = time.time()
            for tokens in sequences:
                (address, tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, rng)
                instance = TrainingInstance(
                    address=address,
                    tokens=tokens,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            end = time.time()
            cost = end - start
            print("step=%d, time=%.2f" % (step, cost))
        print("=======Finish========")
    return instances


def create_masked_lm_predictions_force_head(tokens):
    """Creates the predictions for the masked LM objective."""
    first_index = 0
    address = tokens[0][0]
    output_tokens = [list(i) for i in tokens]  # note that change the value of output_tokens will also change tokens
    output_tokens[first_index] = ["[MASK]", 0, 0, 0, 0, 0]
    masked_lm_positions = [first_index]
    masked_lm_labels = [tokens[first_index][0]]

    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, rng):
    """Creates the predictions for the masked LM objective."""

    address = tokens[0][0]
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = [list(i) for i in tokens]  # note that change the value of output_tokens will also change tokens
    # num_to_predict = min(max_predictions_per_seq,
    #                      max(1, int(round(len(tokens) * masked_lm_prob))))
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(len(tokens) * masked_lm_prob)))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = "[MASK]"
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index][0]))
        output_tokens[index][0] = masked_token

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def create_embedding_predictions(tokens):
    """Creates the predictions for the masked LM objective."""
    address = tokens[0][0]
    output_tokens = tokens
    masked_lm_positions = []
    masked_lm_labels = []
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def gen_embedding_samples(sequences):
    instances = []
    # create train
    start = time.time()
    for tokens in sequences:
        (address, tokens, masked_lm_positions,
         masked_lm_labels) = create_embedding_predictions(tokens)
        instance = TrainingInstance(
            address=address,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances


def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    for inst_index in tqdm(range(len(instances))):
        instance = instances[inst_index]
        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        address = vocab.convert_tokens_to_ids(instance.address)
        counts = instance.cnts
        block_timestamps = instance.block_timestamps
        values = instance.cnts
        io_flags = instance.io_flags
        positions = convert_timestamp_to_position(block_timestamps)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["address"] = create_int_feature(address)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_positions"] = create_int_feature(positions)
        features["input_counts"] = create_int_feature(counts)
        features["input_io_flags"] = create_int_feature(io_flags)
        features["input_values"] = create_int_feature(values)

        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


# def feature_statistics(eoa_keys_split, eoa2account, step):
#     eoa2seq_sub = {}
#     cnt = 0
#     start_time = time.time()
#     for eoa in eoa_keys_split[step]:
#         cnt += 1
#         if cnt % 1000 == 0:
#             print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt, time.time() - start_time))
#             start_time = time.time()
#         df = pd.DataFrame(eoa2account[eoa], columns=["account", "block_number", "block_timestamp", "value", "io_flag", "cnt"])
#         eoa2seq_sub[eoa] = []
#         for idx, row in df.iterrows():
#             eoa2seq_sub[eoa].append([row.account, int(row.block_timestamp), row.cnt])
#     return eoa2seq_sub

def main():
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open("./data/eoa2seq_" + FLAGS.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename + "." + FLAGS.bizdate
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    # eoa_keys = list(eoa2seq.keys()) # split into pool_size lists
    # split_range = round(len(eoa_keys)/FLAGS.pool_size)
    # eoa_keys_split = [eoa_keys[i*split_range:(i+1)*split_range] for i in range(FLAGS.pool_size)]
    # eoa2seq = {}
    # def log_result(result):
    #     print("callback function result type: {}, size: {} ".format(type(result), len(result)))
    #     eoa2seq.update(result)
    #
    # pool = multiprocessing.Pool(processes=FLAGS.pool_size)
    # for step in range(FLAGS.pool_size):
    #     # eoa2seq_sub = feature_statistics(eoa_keys_split, step)
    #     # eoa2seq.update(eoa2seq_sub)
    #     pool.apply_async(feature_statistics, args=(eoa_keys_split, eoa2seq, step), callback=log_result)
    #
    # pool.close()
    # pool.join()
    # for i in tqdm(range(len(eoa_keys))):
    #     eoa = eoa_keys[i]
    #     df = pd.DataFrame(eoa2account[eoa], columns=["account", "block_timestamp"])
    #     df_new = df.groupby(["account","block_timestamp"]).size().reset_index(name="cnt").sort_values("block_timestamp", ascending=False)
    #     eoa2seq[eoa] = []
    #     for idx, row in df_new.iterrows():
    #         eoa2seq[eoa].append([row.account, int(row.block_timestamp), row.cnt])

    # clip
    max_num_tokens = FLAGS.max_seq_length - 1
    seqs = []
    idx = 0
    for eoa, seq in eoa2seq.items():
        if len(seq) <= max_num_tokens:
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq
            idx += 1
        elif len(seq) > max_num_tokens:
            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * FLAGS.sliding_step))
            beg_idx.append(0)

            if len(beg_idx) > 500:
                beg_idx = list(np.random.permutation(beg_idx)[:500])
                for i in beg_idx:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

            else:
                for i in beg_idx[::-1]:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

    if FLAGS.do_embed:
        print("===========Generate Embedding Samples==========")
        write_instance = gen_embedding_samples(seqs)
        output_filename = FLAGS.data_dir + FLAGS.dataset_name + ".embed.tfrecord" + "." + FLAGS.bizdate
        tf.logging.info("*** Writing to output embedding files ***")
        tf.logging.info("  %s", output_filename)

        write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                        FLAGS.max_predictions_per_seq, vocab,
                                        [output_filename])

    seqs = np.random.permutation(seqs)

    if FLAGS.do_eval:  # select 20% for testing
        print("========Generate Evaluation Samples========")
        eval_seqs = seqs[:round(len(seqs) * 0.2)]
        seqs = seqs[round(len(seqs) * 0.2):]

        eval_normal_instances = gen_samples(eval_seqs,
                                            dupe_factor=FLAGS.dupe_factor,
                                            masked_lm_prob=FLAGS.masked_lm_prob,
                                            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                            pool_size=FLAGS.pool_size,
                                            rng=rng,
                                            force_head=False)

        # eval_force_head_sample = gen_samples(eval_seqs,
        #                                      dupe_factor=FLAGS.dupe_factor,
        #                                      masked_lm_prob=FLAGS.masked_lm_prob,
        #                                      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        #                                      pool_size=FLAGS.pool_size,
        #                                      rng=rng,
        #                                      force_head=True)
        eval_force_head_sample = []

        eval_write_instance = eval_normal_instances + eval_force_head_sample
        rng.shuffle(eval_write_instance)
        eval_output_filename = FLAGS.data_dir + FLAGS.dataset_name + ".test.tfrecord" + "." + FLAGS.bizdate
        tf.logging.info("*** Writing to Testing files ***")
        tf.logging.info("  %s", eval_output_filename)

        write_instance_to_example_files(eval_write_instance, FLAGS.max_seq_length,
                                        FLAGS.max_predictions_per_seq, vocab,
                                        [eval_output_filename])

    print("========Generate Training Samples========")
    normal_instances = gen_samples(seqs,
                                   dupe_factor=FLAGS.dupe_factor,
                                   masked_lm_prob=FLAGS.masked_lm_prob,
                                   max_predictions_per_seq=FLAGS.max_predictions_per_seq,
                                   pool_size=FLAGS.pool_size,
                                   rng=rng,
                                   force_head=False)

    # force_head_sample = gen_samples(seqs,
    #                                 dupe_factor=FLAGS.dupe_factor,
    #                                 masked_lm_prob=FLAGS.masked_lm_prob,
    #                                 max_predictions_per_seq=FLAGS.max_predictions_per_seq,
    #                                 pool_size=FLAGS.pool_size,
    #                                 rng=rng,
    #                                 force_head=True)
    force_head_sample = []

    write_instance = normal_instances + force_head_sample
    rng.shuffle(write_instance)

    output_filename = FLAGS.data_dir + FLAGS.dataset_name + ".train.tfrecord" + "." + FLAGS.bizdate
    tf.logging.info("*** Writing to Training files ***")
    tf.logging.info("  %s", output_filename)

    write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, vocab,
                                    [output_filename])

    return

if __name__ == '__main__':
    main()


