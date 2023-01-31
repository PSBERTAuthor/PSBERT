import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import sys
sys.path.append("../BERT")
import modeling

from sklearn import metrics
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("metric", None, "")
flags.DEFINE_string("ens_dataset", "../../Data/dean_all_ens_pairs.csv", "")
flags.DEFINE_integer("max_cnt", 2, "")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("cross", True, "whether use residual mlp")
flags.DEFINE_string("bizdate", None, "the signature for dataset")

if FLAGS.bizdate is None:
    raise ValueError("bizdate is required..")

def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

def cosine_dist(a, b):
    return 1-dot(a, b)/(norm(a)*norm(b)) # notice, here need 1-

def cosine_dist_multi(a, b):
    num = dot(a, b.T)
    denom = norm(a) * norm(b, axis=1)
    res = num/denom
    return -1 * res

def euclidean_dist_multi(a, b):
    return np.sqrt(np.sum(np.square(b-a), axis=1))

def load_mlp_from_checkpoint(init_checkpoint):

    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)

    return init_checkpoint, assignment_map

def DNN(src_embeddings, dst_emheddings):

    sequence_shape = modeling.get_shape_list(src_embeddings, expected_rank=2)
    hidden_size = sequence_shape[1]

    with tf.variable_scope("MLP", reuse=tf.AUTO_REUSE):
        if FLAGS.cross:

            residual_inp = tf.abs(src_embeddings - dst_emheddings)
            multiply_inp = src_embeddings * dst_emheddings
            inp = tf.concat([src_embeddings, dst_emheddings, residual_inp, multiply_inp], axis=1)

        else:
            print("No cross aware!!")
            inp = tf.concat([src_embeddings, dst_emheddings], axis=1)

        dnn1 = tf.layers.dense(inp, hidden_size, activation=tf.nn.relu, name="f1")
        dnn2 = tf.layers.dense(inp, hidden_size, activation=tf.nn.relu, name="f2")
        logit = tf.squeeze(tf.layers.dense(dnn2+dnn1, 1, activation=None, name="logit"))
        logit = tf.sigmoid(logit)

    return logit

def classifier_dist_multi(a, b):

    src_embeddings = tf.constant(np.tile(a, [len(b), 1]))
    dst_emheddings = tf.constant(b)

    logit = DNN(src_embeddings, dst_emheddings)
    init_checkpoint, assignment_map = load_mlp_from_checkpoint(FLAGS.init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, FLAGS.init_checkpoint)
        y_pred_proba = sess.run(logit)

    tf.reset_default_graph()
    return -1 * y_pred_proba

def get_neighbors(X, idx, metric="cosine" ,include_idx_mask=[]):
    a = X[idx, :]
    indices = list(range(X.shape[0]))
    if metric == "cosine":
        # dist = np.array([cosine_dist(a, X[i, :]) for i in indices])
        dist = cosine_dist_multi(a, X)
    elif metric == "euclidean":
        dist = euclidean_dist_multi(a, X)
    elif metric == "mlp":
        dist = classifier_dist_multi(a, X)
    else:
        raise ValueError("Distance Metric Error")
    sorted_df = pd.DataFrame(list(zip(indices, dist)), columns=["idx", "dist"]).sort_values("dist")
    sorted_df = sorted_df.drop(index=idx)  # exclude self distance
    indices = list(sorted_df["idx"])
    distances = list(sorted_df["dist"])

    if len(include_idx_mask) > 0:
        # filter indices
        indices_tmp = []
        distances_tmp = []
        for i, res_idx in enumerate(indices):
            if res_idx in include_idx_mask:
                indices_tmp.append(res_idx)
                distances_tmp.append(distances[i])
        indices = indices_tmp
        distances = distances_tmp
    return indices, distances


def get_rank(X, query_idx, target_idx, metric, include_idx_mask=[]):
    indices, distances = get_neighbors(X, query_idx, metric, include_idx_mask)
    if len(indices) > 0 and target_idx in indices:
        trg_idx = indices.index(target_idx)
        return trg_idx+1, distances[trg_idx], len(indices)
    else:
        return None, None, len(indices)

def generate_pairs(ens_pairs, min_cnt=2, max_cnt=2, mirror=True):
    """
    Generate testing pairs based on ENS name
    :param ens_pairs:
    :param min_cnt:
    :param max_cnt:
    :param mirror:
    :return:
    """
    pairs = ens_pairs.copy()
    ens_counts = pairs["name"].value_counts()
    address_pairs = []
    all_ens_names = []
    ename2addresses = {}
    for idx, row in pairs.iterrows():
        try:
            ename2addresses[row["name"]].append(row["address"]) # note: cannot use row.name
        except:
            ename2addresses[row["name"]] = [row["address"]]
    for cnt in range(min_cnt, max_cnt + 1):
        ens_names = list(ens_counts[ens_counts == cnt].index)
        all_ens_names += ens_names
        # convert to indices
        for ename in ens_names:
            addrs = ename2addresses[ename]
            for i in range(len(addrs)):
                for j in range(i + 1, len(addrs)):
                    addr1, addr2 = addrs[i], addrs[j]
                    address_pairs.append([addr1, addr2])
                    if mirror:
                        address_pairs.append([addr2, addr1])
    return address_pairs, all_ens_names

if __name__ == '__main__':
    # load pair
    ens_pairs = pd.read_csv(FLAGS.ens_dataset)
    max_ens_per_address = 1
    num_ens_for_addr = ens_pairs.groupby("address")["name"].nunique().sort_values(ascending=False).reset_index()
    excluded = list(num_ens_for_addr[num_ens_for_addr["name"] > max_ens_per_address]["address"])
    ens_pairs = ens_pairs[~ens_pairs["address"].isin(excluded)]
    address_pairs, all_ens_names = generate_pairs(ens_pairs, max_cnt=FLAGS.max_cnt)

    # load embedding

    ckpt_dir = str(FLAGS.init_checkpoint.split("/")[0])
    model_index = str(FLAGS.init_checkpoint.split("/")[-1].split("_")[1])

    embedding_file_name = "./data/bert_siamese_deself_embedding_" + ckpt_dir + "_" + model_index + "_" + FLAGS.bizdate + ".npy"
    address_file_name = "./data/address_bert_siamese_" + ckpt_dir + "_" + model_index + "_" + FLAGS.bizdate + ".npy"

    print("load embedding from: ", embedding_file_name)
    print("load address from: ", address_file_name)

    embeddings = np.load(embedding_file_name)
    address_for_embedding = np.load(address_file_name)

    # group by embedding according to address
    address_to_embedding = {}
    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        # if address not in exp_addr_set:
        #     continue
        try:
            address_to_embedding[address].append(embedding)
        except:
            address_to_embedding[address] = [embedding]

    # group to one
    address_list = []
    embedding_list = []

    for addr, embeds in address_to_embedding.items():
        address_list.append(addr)
        if len(embeds) > 1:
            embedding_list.append(np.mean(embeds, axis=0))
        else:
            embedding_list.append(embeds[0])

    # final embedding table
    embedding_list = np.array(np.squeeze(embedding_list))

    # map address to int
    cnt = 0
    address_to_idx = {}
    idx_to_address = {}
    for address in address_list:
        address_to_idx[address] = cnt
        idx_to_address[cnt] = address
        cnt += 1

    idx_pairs = []
    failed_address = []
    for pair in address_pairs:
        try:
            idx_pairs.append([address_to_idx[pair[0]], address_to_idx[pair[1]]])
        except:
            failed_address.append(pair[0])
            failed_address.append(pair[1])
            continue

    pbar = tqdm(total=len(idx_pairs))
    records = []
    X = embedding_list

    for pair in idx_pairs:
        rank, dist, num_set = get_rank(X, pair[1], pair[0], FLAGS.metric)
        records.append((pair[1], pair[0], rank, dist, num_set, "none"))
        print(rank)
        pbar.update(1)

    result = pd.DataFrame(records, columns=["query_idx", "target_idx", "rank", "dist", "set_size", "filter"])
    result["query_addr"] = result["query_idx"].apply(lambda x: idx_to_address[x])
    result["target_addr"] = result["target_idx"].apply(lambda x: idx_to_address[x])
    result.drop(["query_idx", "target_idx"], axis=1)

    output_file = "./data/bert_siamese_deself_embedding_" + ckpt_dir + "_" + model_index + "_" + FLAGS.bizdate + "_euclidean.csv"
    print("saving result to: ", output_file)
    result.to_csv(output_file, index=False)