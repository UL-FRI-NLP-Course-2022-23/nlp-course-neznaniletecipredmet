import argparse
import json
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from transformers import BertModel, BertTokenizer


import logging
import os
import time

from tqdm import tqdm

# import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import get_clusters
# from visualization import build_and_display

import json
from collections import Counter
import logging
import os
from typing import List, Optional, Mapping

from sklearn.model_selection import train_test_split


PAD_TOKEN, PAD_ID = "<PAD>", 0
BOS_TOKEN, BOS_ID = "<BOS>", 1
EOS_TOKEN, EOS_ID = "<EOS>", 2
UNK_TOKEN, UNK_ID = "<UNK>", 3

# from common import ControllerBase, NeuralCoreferencePairScorer
# from data import read_corpus, Document
# from utils import split_into_sets, fixed_split, KFoldStateCache

import os
import logging
import csv
import pandas as pd
from collections import OrderedDict

from bs4 import BeautifulSoup

import neleval.coref_metrics as metrics

DUMMY_ANTECEDENT = None
#
# #####################
# # GLOBAL PARAMETERS
# #####################
# # Path "./data/*" assumes you are running from root folder, i.e. (python /src/baseline.py)
# # Use path "../data/*" if you are running from src folder, i.e. (cd src) and then (python baseline.py)
COREF149_DIR = os.environ.get("COREF149_DIR", "slo_coref/data/coref149")
SENTICOREF_DIR = os.environ.get("SENTICOREF149_DIR", "./data/senticoref1_0")
SENTICOREF_METADATA_DIR = "./data/senticoref_pos_stanza"
SSJ_PATH = os.environ.get("SSJ_PATH", "slo_coref/data/ssj500k-sl.TEI/ssj500k-sl.body.reduced.xml")

class Score:

    def __init__(self):
        # precision, recall, f1
        self.prf = (0.0, 0.0, 0.0)
        self._add_count = 0
        pass

    def precision(self):
        return self.prf[0] / self._add_count

    def recall(self):
        return self.prf[1] / self._add_count

    def f1(self):
        return self.prf[2] / self._add_count

    def add(self, prf):
        # prf = tuple of (precision, recall, F1)
        # usually as a result of some metric function below (muc, b_cubed, ceaf_e)
        self.prf = (
                self.prf[0] + prf[0],
                self.prf[1] + prf[1],
                self.prf[2] + prf[2],
        )
        self._add_count += 1  # used for normalization

    def __str__(self):
        return f"prec={self.precision():.3f}, rec={self.recall():.3f}, f1={self.f1():.3f}"


def conll_12(muc_score, b_cubed_score, ceaf_e_score):
    # CoNLL-12 metric is an average of MUC, B3 and CEAF metric.
    s = Score()
    s.add((muc_score.precision(), muc_score.recall(), muc_score.f1()))
    s.add((b_cubed_score.precision(), b_cubed_score.recall(), b_cubed_score.f1()))
    s.add((ceaf_e_score.precision(), ceaf_e_score.recall(), ceaf_e_score.f1()))
    return s


def muc(gold, resp):
    return metrics._prf(*metrics.muc(gold, resp))


def b_cubed(gold, resp):
    return metrics._prf(*metrics.b_cubed(gold, resp))


def ceaf_e(gold, resp):
    return metrics._prf(*metrics.entity_ceaf(gold, resp))

class KFoldStateCache:
    def __init__(self, script_name: str, main_dataset: str, fold_info: List[dict],
                 additional_dataset: Optional[str] = None,
                 script_args: Optional[Mapping] = None):
        self.script_name = script_name
        self.fold_info = fold_info
        self.num_folds = len(self.fold_info)

        self.script_args = script_args if script_args is not None else {}

        # The dataset that is being split with KFold CV
        self.main_dataset = main_dataset
        # For combined runners: documents, read with `read_corpus(additional_dataset)` should be placed in training set
        self.additional_dataset = additional_dataset

    def get_next_unfinished(self):
        for i, curr_fold in enumerate(self.fold_info):
            if curr_fold.get("results", None) is None:
                yield {
                    "idx_fold": i,
                    "train_docs": curr_fold["train_docs"],
                    "test_docs": curr_fold["test_docs"]
                }

    def add_results(self, idx_fold, results):
        self.fold_info[idx_fold]["results"] = results

    def save(self, path):
        _path = path if path.endswith(".json") else f"{path}.json"
        if os.path.exists(_path):
            logging.warning(f"Overwriting KFold cache at '{_path}'")
        with open(_path, "w", encoding="utf8") as f:
            json.dump({
                "script_name": self.script_name,
                "script_args": self.script_args,
                "main_dataset": self.main_dataset,
                "additional_dataset": self.additional_dataset,
                "fold_info": self.fold_info
            }, fp=f, indent=4)

    @staticmethod
    def from_file(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = KFoldStateCache(script_name=data["script_name"],
                                   script_args=data.get("script_args", None),
                                   main_dataset=data["main_dataset"],
                                   fold_info=data["fold_info"],
                                   additional_dataset=data.get("additional_dataset", None))
        return instance


def extract_vocab(documents, top_n=10_000, lowercase=False):
    token_counter = Counter()
    for curr_doc in documents:
        curr_sentences = curr_doc.raw_sentences()

        for sent_tokens in curr_sentences:
            processed = list(map(lambda s: s.lower() if lowercase else s, sent_tokens))
            token_counter += Counter(processed)

    tok2id, id2tok = {}, {}
    special_tokens = [(PAD_TOKEN, PAD_ID), (BOS_TOKEN, BOS_ID), (EOS_TOKEN, EOS_ID), (UNK_TOKEN, UNK_ID)]
    for t, i in special_tokens:
        tok2id[t] = i
        id2tok[i] = t

    for i, (token, _) in enumerate(token_counter.most_common(top_n), start=len(special_tokens)):
        tok2id[token] = i
        id2tok[i] = token

    return tok2id, id2tok


def encode(seq, vocab, max_seq_len):
    encoded_seq = []
    for i, curr_token in enumerate(seq):
        encoded_seq.append(vocab.get(curr_token, vocab["<UNK>"]))

    # If longer than max allowed length, truncate sequence; otherwise pad with a special symbol
    if len(seq) > max_seq_len:
        encoded_seq = encoded_seq[: max_seq_len]
    else:
        encoded_seq += [vocab["<PAD>"]] * (max_seq_len - len(seq))

    return encoded_seq

def get_clusters(preds):
    """ Convert {antecedent_id: mention_id} pairs into {mention_id: assigned_cluster_id} pairs. """
    cluster_assignments = {}

    for id_cluster, cluster_starter in enumerate(preds.get(None, [])):
        stack = [cluster_starter]
        curr_cluster = []
        while len(stack) > 0:
            cur = stack.pop()
            curr_cluster.append(cur)
            cluster_assignments[cur] = id_cluster
            mentions = preds.get(cur)
            if mentions is not None:
                stack.extend(mentions)

    return cluster_assignments


def split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15):
    """
    Splits documents array into three sets: learning, validation & testing.
    If random seed is given, documents selected for each set are randomly picked (but do not overlap, of course).
    """
    # Note: test_prop is redundant, but it's left in to make it clear this is a split into 3 parts
    test_prop = 1.0 - train_prop - dev_prop

    train_docs, dev_test_docs = train_test_split(documents, test_size=(dev_prop + test_prop))

    dev_docs, test_docs = train_test_split(dev_test_docs, test_size=test_prop/(dev_prop + test_prop))

    logging.info(f"{len(documents)} documents split to: training set ({len(train_docs)}), dev set ({len(dev_docs)}) "
                 f"and test set ({len(test_docs)}).")

    return train_docs, dev_docs, test_docs


def fixed_split(documents, dataset):
    tr, dev, te = read_splits(os.path.join("", "data", "seeded_split", f"{dataset}.txt"))
    assert (len(tr) + len(dev) + len(te)) == len(documents)

    train_docs = list(filter(lambda doc: doc.doc_id in tr, documents))
    dev_docs = list(filter(lambda doc: doc.doc_id in dev, documents))
    te_docs = list(filter(lambda doc: doc.doc_id in te, documents))
    return train_docs, dev_docs, te_docs


def read_splits(file_path):
    with open(file_path, "r") as f:
        doc_ids = []
        # train, dev, test
        for _ in range(3):
            curr_ids = set(f.readline().strip().split(","))
            doc_ids.append(curr_ids)

        return doc_ids


class ControllerBase:
    def __init__(self, learning_rate, dataset_name, early_stopping_rounds=5, model_name=None):
        self.model_name = time.strftime("%Y%m%d_%H%M%S") if model_name is None else model_name
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds

        # Mention ranking model = always using cross-entropy
        self.loss = nn.CrossEntropyLoss()

        # Put various debugging/visualization related things in model dir
        self.path_model_dir = os.path.join(self.model_base_dir, self.model_name)
        self.path_metadata = os.path.join(self.path_model_dir, "model_metadata.txt")
        self.path_pred_clusters = os.path.join(self.path_model_dir, "pred_clusters.txt")
        self.path_pred_scores = os.path.join(self.path_model_dir, "pred_scores.txt")
        self.path_log = os.path.join(self.path_model_dir, "log.txt")

        self.loaded_from_file = False
        # self._prepare()

    @property
    def model_base_dir(self):
        """ Should return the directory where models of this type should be saved. """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(model_dir):
        raise NotImplementedError

    def save_pretrained(self, model_dir):
        raise NotImplementedError

    def _prepare(self):
        if os.path.exists(self.path_model_dir):
            self.load_checkpoint()
        else:
            os.makedirs(self.path_model_dir)
            logger.addHandler(logging.FileHandler(self.path_log, mode="w", encoding="utf-8"))
            logging.info(f"Created directory '{self.path_model_dir}' for model files")

    def load_checkpoint(self):
        """ Should load weights and other checkpoint-related data for underlying model of controller. """
        raise NotImplementedError

    def save_checkpoint(self):
        """ Should save weights and other checkpoint-related data for underlying model of controller. """
        pass

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document. Returns predictions, loss and number
            of examples evaluated. """
        raise NotImplementedError

    def train_mode(self):
        """ Set underlying modules to train mode. """
        raise NotImplementedError

    def eval_mode(self):
        """ Set underlying modules to eval mode. """
        raise NotImplementedError

    def train(self, epochs, train_docs, dev_docs):
        logging.info("Starting training")

        best_dev_loss, best_epoch = float("inf"), None
        t_start = time.time()
        for idx_epoch in range(epochs):
            t_epoch_start = time.time()
            shuffle_indices = torch.randperm(len(train_docs))

            self.train_mode()
            train_loss, train_examples = 0.0, 0
            for idx_doc in tqdm(shuffle_indices):
                curr_doc = train_docs[idx_doc]

                _, (doc_loss, n_examples) = self._train_doc(curr_doc)

                train_loss += doc_loss
                train_examples += n_examples

            self.eval_mode()
            dev_loss, dev_examples = 0.0, 0
            for curr_doc in dev_docs:
                _, (doc_loss, n_examples) = self._train_doc(curr_doc, eval_mode=True)

                dev_loss += doc_loss
                dev_examples += n_examples

            train_loss /= max(1, train_examples)
            dev_loss /= max(1, dev_examples)

            logging.info(f"[Epoch #{1 + idx_epoch}] "
                         f"training loss: {train_loss: .4f}, dev loss: {dev_loss: .4f} "
                         f"[took {time.time() - t_epoch_start:.2f}s]")

            if dev_loss < best_dev_loss:
                self.save_checkpoint()
                # Save this score as best
                best_dev_loss = dev_loss
                best_epoch = idx_epoch

            if idx_epoch - best_epoch == self.early_stopping_rounds:
                logging.info(f"Validation metric did not improve for {self.early_stopping_rounds} rounds, "
                             f"stopping early")
                break

        logging.info(f"Training complete: took {time.time() - t_start:.2f}s")

        # Add model train scores to model metadata
        with open(self.path_metadata, "a", encoding="utf-8") as f:
            logging.info(f"Saving best validation score to {self.path_metadata}")
            f.writelines([
                "\n",
                "Train model scores:\n",
                f"Best validation set loss: {best_dev_loss}\n",
            ])

        return best_dev_loss

    def evaluate_single(self, document):
        # doc_name: <cluster assignments> pairs for all test documents
        logging.info("Evaluating a single document...")

        predictions, _, probabilities = self._train_doc(document, eval_mode=True)
        clusters = get_clusters(predictions)
        scores = {m: probabilities[m] for m in clusters.keys()}

        return { "predictions": predictions, "clusters": clusters, "scores": scores }

    @torch.no_grad()
    def evaluate(self, test_docs):
        # doc_name: <cluster assignments> pairs for all test documents
        logging.info("Evaluating...")
        all_test_preds = {}

        # [MUC score]
        # The MUC score counts the minimum number of links between mentions
        # to be inserted or deleted when mapping a system response to a gold standard key set
        # [B3 score]
        # B3 computes precision and recall for all mentions in the document,
        # which are then combined to produce the final precision and recall numbers for the entire output
        # [CEAF score]
        # CEAF applies a similarity metric (either mention based or entity based) for each pair of entities
        # (i.e. a set of mentions) to measure the goodness of each possible alignment.
        # The best mapping is used for calculating CEAF precision, recall and F-measure
        muc_score = metrics.Score()
        b3_score = metrics.Score()
        ceaf_score = metrics.Score()

        for curr_doc in tqdm(test_docs):

            test_preds, _ = self._train_doc(curr_doc, eval_mode=True)
            test_clusters = get_clusters(test_preds)

            # Save predicted clusters for this document id
            all_test_preds[curr_doc.doc_id] = test_clusters

            # input into metric functions should be formatted as dictionary of {int -> set(str)},
            # where keys (ints) are clusters and values (string sets) are mentions in a cluster. Example:
            # {
            #  1: {'rc_1', 'rc_2', ...}
            #  2: {'rc_5', 'rc_8', ...}
            #  3: ...
            # }

            # gt = ground truth, pr = predicted by model
            gt_clusters = {k: set(v) for k, v in enumerate(curr_doc.clusters)}
            pr_clusters = {}
            for (pr_ment, pr_clst) in test_clusters.items():
                if pr_clst not in pr_clusters:
                    pr_clusters[pr_clst] = set()
                pr_clusters[pr_clst].add(pr_ment)

            muc_score.add(metrics.muc(gt_clusters, pr_clusters))
            b3_score.add(metrics.b_cubed(gt_clusters, pr_clusters))
            ceaf_score.add(metrics.ceaf_e(gt_clusters, pr_clusters))

        avg_score = metrics.conll_12(muc_score, b3_score, ceaf_score)
        logging.info(f"----------------------------------------------")
        logging.info(f"**Test scores**")
        logging.info(f"**MUC:      {muc_score}**")
        logging.info(f"**BCubed:   {b3_score}**")
        logging.info(f"**CEAFe:    {ceaf_score}**")
        logging.info(f"**CoNLL-12: {avg_score}**")
        logging.info(f"----------------------------------------------")

        # Save test predictions and scores to file for further debugging
        with open(self.path_pred_scores, "w", encoding="utf-8") as f:
            f.writelines([
                f"Database: {self.dataset_name}\n\n",
                f"Test scores:\n",
                f"MUC:      {muc_score}\n",
                f"BCubed:   {b3_score}\n",
                f"CEAFe:    {ceaf_score}\n",
                f"CoNLL-12: {metrics.conll_12(muc_score, b3_score, ceaf_score)}\n",
            ])
        with open(self.path_pred_clusters, "w", encoding="utf-8") as f:
            f.writelines(["Predictions:\n"])
            for doc_id, clusters in all_test_preds.items():
                f.writelines([
                    f"Document '{doc_id}':\n",
                    str(clusters), "\n"
                ])

        return {
            "muc": muc_score,
            "b3": b3_score,
            "ceafe": ceaf_score,
            "avg": avg_score
        }

    # def visualize(self):
    #     build_and_display(self.path_pred_clusters, self.path_pred_scores, self.path_model_dir, display=False)


class NeuralCoreferencePairScorer(nn.Module):
    def __init__(self, num_features, hidden_size=150, dropout=0.2):
        # Note: num_features is either hidden_size of a LSTM or 2*hidden_size if using biLSTM
        super().__init__()

        # Attempts to model head word (root) in a mention, e.g. "model" in "my amazing model"
        self.attention_projector = nn.Linear(in_features=num_features, out_features=1)
        self.dropout = nn.Dropout(p=dropout)

        # Converts [candidate_state, head_state, candidate_state * head_state] into a score
        # self.fc = nn.Linear(in_features=(3 * num_features) * 3, out_features=1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=((3 * num_features) * 3), out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=1)
        )

    def forward(self, candidate_features, head_features,
                candidate_attention_mask=None,
                head_attention_mask=None):
        """

        Args:
            candidate_features: [B, num_tokens_cand, num_features]
            head_features: [B, num_tokens_head, num_features]
        """
        eff_cand_attn = candidate_attention_mask.bool() if candidate_attention_mask is not None \
            else torch.ones(candidate_features.shape[:2], dtype=torch.bool)
        eff_head_attn = head_attention_mask.bool() if head_attention_mask is not None \
            else torch.ones(head_features.shape[:2], dtype=torch.bool)

        candidate_features[torch.logical_not(eff_cand_attn)] = 0.0
        head_features[torch.logical_not(eff_head_attn)] = 0.0

        candidate_lengths = torch.sum(eff_cand_attn, dim=1)
        head_lengths = torch.sum(eff_head_attn, dim=1)

        batch_index = torch.arange(candidate_features.shape[0], device=candidate_features.device)

        # Create candidate representation
        candidate_attn_weights = F.softmax(self.attention_projector(self.dropout(candidate_features)),
                                           dim=1)
        cand_attended_features = torch.sum(candidate_attn_weights * candidate_features, dim=1)
        candidate_repr = torch.cat((candidate_features[:, 0],  # first word of mention
                                    candidate_features[batch_index, candidate_lengths - 1],  # last word of mention
                                    cand_attended_features), dim=1)

        # Create head mention representation
        head_attn_weights = F.softmax(self.attention_projector(self.dropout(head_features)),
                                      dim=1)
        head_attended_features = torch.sum(head_attn_weights * head_features, dim=1)
        head_repr = torch.cat((head_features[:, 0],  # first word of mention
                               head_features[batch_index, head_lengths - 1],  # last word of mention
                               head_attended_features), dim=1)

        # Combine representations and compute a score
        pair_score = self.fc(self.dropout(torch.cat((candidate_repr,
                                                     head_repr,
                                                     candidate_repr * head_repr), dim=1)))
        return pair_score


def _read_tokens(corpus_soup):
    """ Obtain all tokens in current document.

    Arguments
    ---------
    corpus_soup: bs4.element.Tag
        Wrapped XML element containing the document (<tc:TextCorpus ...> tag).

    Returns
    -------
    dict[str, str]:
        Mapping of token IDs to raw tokens
    """
    id_to_tok = OrderedDict()
    for i, el in enumerate(corpus_soup.findAll("tc:token")):
        token_id = el["id"]
        token = el.text.strip()
        id_to_tok[token_id] = token
    return id_to_tok


def _read_sentences(corpus_soup):
    """ Obtain all sentences in current document.

    Returns
    -------
    tuple:
        (list[list[str]], dict[str, list]):
            (1.) token IDs, organized into sentences
            (2.) token IDs to [index of sentence, index of token inside sentence]
    """
    sent_tok_ids = []
    tok_to_position = {}
    for idx_sent, el in enumerate(corpus_soup.findAll("tc:sentence")):
        token_ids = el["tokenids"].split(" ")
        for idx_tok, tok in enumerate(token_ids):
            tok_to_position[tok] = [idx_sent, idx_tok]
        sent_tok_ids.append(token_ids)
    return sent_tok_ids, tok_to_position


def _read_coreference(corpus_soup):
    """ Obtain all mentions and coreference clusters in current document.

    Returns
    -------
    tuple:
        (dict[str, list[str]], list[list[str]]):
            (1.) mentions
            (2.) mentions organized by coreference cluster
    """
    mentions = {}
    clusters = []
    for cluster_obj in corpus_soup.findAll("tc:entity"):
        curr_cluster = []
        for mention_obj in cluster_obj.findAll("tc:reference"):
            mention_id = mention_obj["id"]
            mention_tokens = mention_obj["tokenids"].split(" ")
            mentions[mention_id] = mention_tokens
            curr_cluster.append(mention_id)

        clusters.append(curr_cluster)
    return mentions, clusters


# Create a dictionary where each mention points to its antecedent (or the dummy antecedent)
def _coreference_chain(clusters_list):
    mapped_clusters = {}
    for curr_cluster in clusters_list:
        for i, curr_mention in enumerate(curr_cluster):
            mapped_clusters[curr_mention] = DUMMY_ANTECEDENT if i == 0 else curr_cluster[i - 1]
    return mapped_clusters


class Token:
    def __init__(self, token_id, raw_text, lemma, msd, sentence_index, position_in_sentence, position_in_document):
        self.token_id = token_id

        self.raw_text = raw_text
        self.lemma = lemma
        self.msd = msd

        self.sentence_index = sentence_index
        self.position_in_sentence = position_in_sentence
        self.position_in_document = position_in_document

        self.gender = self._extract_gender(msd)
        self.number = self._extract_number(msd)
        self.category = msd[0]

    def __str__(self):
        return f"Token(\"{self.raw_text}\")"

    def _extract_number(self, msd_string):
        number = None
        if msd_string[0] == "S" and len(msd_string) >= 4:  # noun/samostalnik
            number = msd_string[3]
        elif msd_string[0] == "G" and len(msd_string) >= 6:  # verb/glagol
            number = msd_string[5]
        # P = adjective (pridevnik), Z = pronoun (zaimek), K = numeral (števnik)
        elif msd_string[0] in {"P", "Z", "K"} and len(msd_string) >= 5:
            number = msd_string[4]

        return number

    def _extract_gender(self, msd_string):
        gender = None
        if msd_string[0] == "S" and len(msd_string) >= 3:  # noun/samostalnik
            gender = msd_string[2]
        elif msd_string[0] == "G" and len(msd_string) >= 7:  # verb/glagol
            gender = msd_string[6]
        # P = adjective (pridevnik), Z = pronoun (zaimek), K = numeral (števnik)
        elif msd_string[0] in {"P", "Z", "K"} and len(msd_string) >= 4:
            gender = msd_string[3]

        return gender


class Mention:
    def __init__(self, mention_id, tokens):
        self.mention_id = mention_id
        self.tokens = tokens

    def __str__(self):
        return f"Mention(\"{' '.join([tok.raw_text for tok in self.tokens])}\")"

    def raw_text(self):
        return " ".join([t.raw_text for t in self.tokens])

    def lemma_text(self):
        return " ".join([t.lemma for t in self.tokens if t.lemma is not None])


class Document:
    def __init__(self, doc_id, tokens, sentences, mentions, clusters,
                 metadata=None):
        self.doc_id = doc_id  # type: str
        self.tokens = tokens  # type: dict
        self.sents = sentences  # type: list
        self.mentions = mentions  # type: dict
        self.clusters = clusters  # type: list
        self.mapped_clusters = _coreference_chain(self.clusters)
        self.metadata = metadata

    def raw_sentences(self):
        """ Returns list of sentences in document. """
        return [list(map(lambda t: self.tokens[t].raw_text, curr_sent)) for curr_sent in self.sents]

    def __len__(self):
        return len(self.tokens)

    def __str__(self):
        return f"Document('{self.doc_id}', {len(self.tokens)} tokens)"


def sorted_mentions_dict(mentions):
    # sorted() produces an array of (key, value) tuples, which we turn back into dictionary
    sorted_mentions = dict(sorted(mentions.items(),
                                  key=lambda tup: (tup[1].tokens[0].sentence_index,  # sentence
                                                   tup[1].tokens[0].position_in_sentence,  # start pos
                                                   tup[1].tokens[-1].position_in_sentence)))  # end pos

    return sorted_mentions


def read_senticoref_doc(file_path):
    # Temporary cluster representation:
    # {cluster1 index: { mention1_idx: ['mention1', 'tokens'], mention2_idx: [...] }, cluster2_idx: {...} }
    _clusters = {}
    # Temporary buffer for current sentence
    _curr_sent = []

    sents = []
    id_to_tok = {}
    tok_to_position = {}
    idx_sent, idx_inside_sent = 0, 0
    mentions, clusters = {}, []

    doc_id = file_path.split(os.path.sep)[-1][:-4]  # = file name without ".tsv"
    # Note: `quoting=csv.QUOTE_NONE` is required as otherwise some documents can't be read
    # Note: `keep_default_na=False` is required as there's a typo in corpus ("NA"), interpreted as <missing>
    curr_annotations = pd.read_table(file_path, comment="#", sep="\t", index_col=False, quoting=csv.QUOTE_NONE,
                                     names=["token_index", "start_end", "token", "NamedEntity", "Polarity",
                                            "referenceRelation", "referenceType"], keep_default_na=False)
    curr_metadata = pd.read_table(os.path.join(SENTICOREF_METADATA_DIR, f"{doc_id}.tsv"), sep="\t", index_col=False,
                                  quoting=csv.QUOTE_NONE, header=0, keep_default_na=False)

    metadata = {"tokens": {}}
    for i, (tok_id, ref_info, token) in enumerate(curr_annotations[["token_index", "referenceRelation", "token"]].values):
        # Token is part of some mention
        if ref_info != "_":
            # Token can be part of multiple mentions
            ref_annotations = ref_info.split("|")

            for mention_info in ref_annotations:
                cluster_idx, mention_idx = list(map(int, mention_info[3:].split("-")))  # skip "*->"

                curr_mentions = _clusters.get(cluster_idx, {})
                curr_mention_tok_ids = curr_mentions.get(mention_idx, [])
                curr_mention_tok_ids.append(tok_id)
                curr_mentions[mention_idx] = curr_mention_tok_ids

                _clusters[cluster_idx] = curr_mentions

        _curr_sent.append(tok_id)
        tok_to_position[tok_id] = [idx_sent, idx_inside_sent]
        id_to_tok[tok_id] = token
        idx_inside_sent += 1

        text, pos_tag, lemma = curr_metadata.iloc[i].values
        metadata["tokens"][tok_id] = {"ana": pos_tag, "lemma": lemma, "text": text}

        # Segment sentences heuristically
        if token in {".", "!", "?"}:
            idx_sent += 1
            idx_inside_sent = 0
            sents.append(_curr_sent)
            _curr_sent = []

    # If the document doesn't end with proper punctuation
    if len(_curr_sent) > 0:
        sents.append(_curr_sent)

    # --- generate token objects
    final_tokens = OrderedDict()
    for index, (tok_id, tok_raw) in enumerate(id_to_tok.items()):
        final_tokens[tok_id] = Token(
            tok_id,
            tok_raw,
            metadata["tokens"][tok_id]["lemma"] if "lemma" in metadata["tokens"][tok_id] else None,
            metadata["tokens"][tok_id]["ana"].split(":")[1],
            tok_to_position[tok_id][0],
            tok_to_position[tok_id][1],
            index
        )
    # ---

    mention_counter = 0
    for idx_cluster, curr_mentions in _clusters.items():
        curr_cluster = []
        for idx_mention, mention_tok_ids in curr_mentions.items():
            # assign coref149-style IDs to mentions
            mention_id = f"rc_{mention_counter}"
            mention_tokens = list(map(lambda tok_id: final_tokens[tok_id], mention_tok_ids))
            mentions[mention_id] = Mention(mention_id, mention_tokens)

            curr_cluster.append(mention_id)
            mention_counter += 1
        clusters.append(curr_cluster)

    return Document(doc_id, final_tokens, sents, sorted_mentions_dict(mentions), clusters, metadata=metadata)


def read_coref149_doc(file_path, ssj_doc):
    with open(file_path, encoding="utf8") as f:
        content = f.readlines()
        content = "".join(content)
        soup = BeautifulSoup(content, "lxml").find("tc:textcorpus")

    doc_id = file_path.split(os.path.sep)[-1][:-4]  # = file name without ".tcf"

    # Read data as defined in coref149
    tokens = _read_tokens(soup)
    sents, tok_to_position = _read_sentences(soup)
    mentions, clusters = _read_coreference(soup)

    # Tokens have different IDs in ssj500k, so remap coref149 style to ssj500k style
    idx_sent_coref, idx_token_coref = 0, 0
    _coref_to_ssj = {} # mapping from coref ids to ssj ids
    for curr_sent in ssj_doc.findAll("s"):
        for curr_token in curr_sent.findAll(["w", "pc"]):
            coref_token_id = sents[idx_sent_coref][idx_token_coref]
            ssj_token_id = curr_token["xml:id"]

            # Warn in case tokenization is different between datasets (we are slightly screwed in that case)
            if curr_token.text.strip() != tokens[coref_token_id]:
                logging.warning(f"MISMATCH! '{curr_token.text.strip()}' (ssj500k ID: {ssj_token_id}) vs "
                                f"'{tokens[coref_token_id]}' (coref149 ID: {coref_token_id})")

            _coref_to_ssj[coref_token_id] = ssj_token_id
            idx_token_coref += 1
            if idx_token_coref == len(sents[idx_sent_coref]):
                idx_sent_coref += 1
                idx_token_coref = 0

    # sentences are composed of ssj token IDs
    fixed_sents = [[_coref_to_ssj[curr_id] for curr_id in curr_sent] for curr_sent in sents]

    # Write all metadata for tokens
    # Note: currently not writing SRL/dependency metadata
    metadata = {"tokens": {}}
    for token in ssj_doc.findAll(["w", "c", "pc"]):
        token_id = token.get("xml:id", None)

        if token_id:
            metadata["tokens"][token_id] = token.attrs
            metadata["tokens"][token_id]["text"] = token.text

    final_tokens = OrderedDict()
    for index, (coref_token_id, raw_text) in enumerate(tokens.items()):
        ssj_token_id = _coref_to_ssj[coref_token_id]  # mapping of coref token ID to ssj token ID
        final_tokens[ssj_token_id] = Token(
            ssj_token_id,
            raw_text,
            metadata["tokens"][ssj_token_id]["lemma"] if "lemma" in metadata["tokens"][ssj_token_id] else None,
            metadata["tokens"][ssj_token_id]["ana"].split(":")[1],
            tok_to_position[coref_token_id][0],  # Note: tok_to_pos uses coref IDs, not ssj IDs
            tok_to_position[coref_token_id][1],
            index)

    final_mentions = {}
    for mention_id, mention_tokens in mentions.items():
        token_objs = [final_tokens[_coref_to_ssj[tok_id]] for tok_id in mention_tokens]
        final_mentions[mention_id] = Mention(mention_id, token_objs)

    # TODO: is metadata required here? metadata for tokens has been moved to token object
    return Document(doc_id, final_tokens, fixed_sents, sorted_mentions_dict(final_mentions), clusters, metadata=metadata)


def read_corpus(name):
    SUPPORTED_DATASETS = {"coref149", "senticoref"}
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset (must be one of {SUPPORTED_DATASETS})")

    if name == "coref149":
        with open(SSJ_PATH, encoding="utf8") as ssj:
            content = ssj.readlines()
            content = "".join(content)
            ssj_soup = BeautifulSoup(content, "lxml")

        doc_to_soup = {}
        for curr_soup in ssj_soup.findAll("p"):
            doc_to_soup[curr_soup["xml:id"]] = curr_soup

        doc_ids = [f[:-4] for f in os.listdir(COREF149_DIR)
                   if os.path.isfile(os.path.join(COREF149_DIR, f)) and f.endswith(".tcf")]
        return [read_coref149_doc(os.path.join(COREF149_DIR, f"{curr_id}.tcf"), doc_to_soup[curr_id]) for curr_id in doc_ids]
    else:
        doc_ids = [f[:-4] for f in os.listdir(SENTICOREF_DIR)
                   if os.path.isfile(os.path.join(SENTICOREF_DIR, f)) and f.endswith(".tsv")]

        return [read_senticoref_doc(os.path.join(SENTICOREF_DIR, f"{curr_id}.tsv")) for curr_id in doc_ids]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--fc_hidden_size", type=int, default=150)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_segment_size", type=int, default=256)
parser.add_argument("--combine_layers", action="store_true",
                    help="Flag to determine if the sequence embeddings should be a learned combination of all "
                         "BERT hidden layers")
parser.add_argument("--dataset", type=str, default="coref149")
parser.add_argument("--pretrained_model_name_or_path", type=str, default="EMBEDDIA/crosloengual-bert")
parser.add_argument("--freeze_pretrained", action="store_true", help="If set, disable updates to BERT layers")
parser.add_argument("--random_seed", type=int, default=13)
parser.add_argument("--fixed_split", action="store_true")
parser.add_argument("--kfold_state_cache_path", type=str, default=None)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class WeightedLayerCombination(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.linear = nn.Linear(embedding_size, out_features=1)

    def forward(self, hidden_states):
        """ Args:
            hidden_states: shape [num_layers, B, seq_len, embedding_size]
        """
        attn_weights = torch.softmax(self.linear(hidden_states), dim=0)  # [num_layers, B, seq_len, 1]
        weighted_combination = torch.sum(attn_weights * hidden_states, dim=0)  # [B, seq_len, embedding_size]

        return weighted_combination, attn_weights


def prepare_document_bert(doc, tokenizer):
    """ Converts a sentence-wise representation of document (list of lists) into a document-wise representation
    (single list) and creates a mapping between the two position indices.

    E.g. a token that is originally in sentence#0 at position#3, might now be broken up into multiple subwords
    at positions [5, 6, 7] in tokenized document."""
    tokenized_doc, mapping = [], {}
    idx_tokenized = 0
    for idx_sent, curr_sent in enumerate(doc.raw_sentences()):
        for idx_inside_sent, curr_token in enumerate(curr_sent):
            tokenized_token = tokenizer.tokenize(curr_token)
            tokenized_doc.extend(tokenized_token)
            mapping[(idx_sent, idx_inside_sent)] = list(range(idx_tokenized, idx_tokenized + len(tokenized_token)))
            idx_tokenized += len(tokenized_token)

    return tokenized_doc, mapping


class ContextualControllerBERT(ControllerBase):
    def __init__(self,
                 dropout,
                 pretrained_model_name_or_path,
                 dataset_name,
                 fc_hidden_size=150,
                 freeze_pretrained=True,
                 learning_rate: float = 0.001,
                 layer_learning_rate: Optional[Dict[str, float]] = None,
                 max_segment_size=512,
                 max_span_size=10,
                 combine_layers=False,
                 model_name=None):
        self.dropout = dropout
        self.fc_hidden_size = fc_hidden_size
        self.freeze_pretrained = freeze_pretrained
        self.max_segment_size = max_segment_size - 3  # CLS, SEP, >= 1 PAD at the end (convention, for batching)
        self.max_span_size = max_span_size
        self.combine_layers = combine_layers
        self.learning_rate = learning_rate
        self.layer_learning_rate = layer_learning_rate if layer_learning_rate is not None else {}

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.embedder = BertModel.from_pretrained(pretrained_model_name_or_path,
                                                  output_hidden_states=combine_layers,
                                                  return_dict=True).to(DEVICE)
        for param in self.embedder.parameters():
            param.requires_grad = not self.freeze_pretrained

        embedding_size = self.embedder.config.hidden_size
        self.combinator = WeightedLayerCombination(embedding_size=embedding_size).to(DEVICE) \
            if self.combine_layers else None
        self.scorer = NeuralCoreferencePairScorer(num_features=embedding_size,
                                                  dropout=dropout,
                                                  hidden_size=fc_hidden_size).to(DEVICE)

        params_to_update = [{
                "params": self.scorer.parameters(),
                "lr": self.layer_learning_rate.get("lr_scorer", self.learning_rate)
        }]
        if not freeze_pretrained:
            params_to_update.append({
                "params": self.embedder.parameters(),
                "lr": self.layer_learning_rate.get("lr_embedder", self.learning_rate)
            })

        if self.combine_layers:
            params_to_update.append({
                "params": self.combinator.parameters(),
                "lr": self.layer_learning_rate.get("lr_combinator", self.learning_rate)
            })

        self.optimizer = optim.Adam(params_to_update, lr=self.learning_rate)

        super().__init__(learning_rate=self.learning_rate, dataset_name=dataset_name, model_name=model_name)
        logging.info(f"Initialized contextual BERT-based model with name {self.model_name}.")

    @property
    def model_base_dir(self):
        return "contextual_model_bert"

    def train_mode(self):
        if not self.freeze_pretrained:
            self.embedder.train()
        if self.combine_layers:
            self.combinator.train()
        self.scorer.train()

    def eval_mode(self):
        self.embedder.eval()
        if self.combine_layers:
            self.combinator.eval()
        self.scorer.eval()

    @staticmethod
    def from_pretrained(model_dir):
        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "r", encoding="utf-8") as f_config:
            pre_config = json.load(f_config)

        # If embeddings are not frozen, they are saved with the controller
        if not pre_config["freeze_pretrained"]:
            pre_config["pretrained_model_name_or_path"] = model_dir

        instance = ContextualControllerBERT(**pre_config)
        instance.path_model_dir = model_dir
        instance.load_checkpoint()

        return instance

    def save_pretrained(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Write controller config (used for instantiation)
        controller_config_path = os.path.join(model_dir, "controller_config.json")
        with open(controller_config_path, "w", encoding="utf-8") as f_config:
            json.dump({
                "dropout": self.dropout,
                "pretrained_model_name_or_path": self.pretrained_model_name_or_path if self.freeze_pretrained else model_dir,
                "dataset_name": self.dataset_name,
                "fc_hidden_size": self.fc_hidden_size,
                "freeze_pretrained": self.freeze_pretrained,
                "learning_rate": self.learning_rate,
                "layer_learning_rate": self.layer_learning_rate,
                "max_segment_size": self.max_segment_size,
                "max_span_size": self.max_span_size,
                "combine_layers": self.combine_layers,
                "model_name": self.model_name
            }, fp=f_config, indent=4)

        torch.save(self.scorer.state_dict(), os.path.join(self.path_model_dir, "scorer.th"))

        # Save fine-tuned BERT embeddings only if they're not frozen
        if not self.freeze_pretrained:
            self.embedder.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)

        if self.combine_layers:
            torch.save(self.combinator.state_dict(), os.path.join(model_dir, "combination.th"))

    def load_checkpoint(self):
        path_to_scorer = os.path.join(self.path_model_dir, "scorer.th")
        self.scorer.load_state_dict(torch.load(path_to_scorer, map_location=DEVICE))
        self.loaded_from_file = True

        if self.combine_layers:
            path_to_combination = os.path.join(self.path_model_dir, "combination.th")
            self.combinator.load_state_dict(torch.load(path_to_combination, map_location=DEVICE))
            self.loaded_from_file = True

    def save_checkpoint(self):
        logging.warning("save_checkpoint() is deprecated. Use save_pretrained() instead")
        self.save_pretrained(self.path_model_dir)

    def _prepare_doc(self, curr_doc: Document) -> Dict:
        """ Returns a cache dictionary with preprocessed data. This should only be called once per document, since
        data inside same document does not get shuffled. """
        ret = {}

        # maps from (idx_sent, idx_token) to (indices_in_tokenized_doc)
        tokenized_doc, mapping = prepare_document_bert(curr_doc, tokenizer=self.tokenizer)
        encoded_doc = self.tokenizer.convert_tokens_to_ids(tokenized_doc)

        num_total_segments = (len(tokenized_doc) + self.max_segment_size - 1) // self.max_segment_size
        segments = {"input_ids": [], "token_type_ids": [], "attention_mask": []}
        idx_token_to_segment = {}
        for idx_segment in range(num_total_segments):
            s_seg, e_seg = idx_segment * self.max_segment_size, (idx_segment + 1) * self.max_segment_size
            for idx_token in range(s_seg, e_seg):
                idx_token_to_segment[idx_token] = (idx_segment, 1 + idx_token - s_seg)  # +1 shift due to [CLS]

            curr_seg = self.tokenizer.prepare_for_model(ids=encoded_doc[s_seg: e_seg],
                                                        max_length=(self.max_segment_size + 2),
                                                        padding="max_length", truncation="longest_first",
                                                        return_token_type_ids=True, return_attention_mask=True)

            # Convention: add a PAD token at the end, so span padding points to that -> [CLS] <segment> [SEP] [PAD]
            segments["input_ids"].append(curr_seg["input_ids"] + [self.tokenizer.pad_token_id])
            segments["token_type_ids"].append(curr_seg["token_type_ids"] + [0])
            segments["attention_mask"].append(curr_seg["attention_mask"] + [0])

        # Shape: [num_segments, (max_segment_size + 3)]
        segments["input_ids"] = torch.tensor(segments["input_ids"])
        segments["token_type_ids"] = torch.tensor(segments["token_type_ids"])
        segments["attention_mask"] = torch.tensor(segments["attention_mask"])

        cluster_sets = []
        mention_to_cluster_id = {}
        for i, curr_cluster in enumerate(curr_doc.clusters):
            cluster_sets.append(set(curr_cluster))
            for mid in curr_cluster:
                mention_to_cluster_id[mid] = i

        all_candidate_data = []
        for idx_head, (head_id, head_mention) in enumerate(curr_doc.mentions.items(), start=1):
            gt_antecedent_ids = cluster_sets[mention_to_cluster_id[head_id]]

            # Note: no data for dummy antecedent (len(`features`) is one less than `candidates`)
            candidates, candidate_data = [None], []
            candidate_attention = []
            correct_antecedents = []

            curr_head_data = [[], []]
            num_head_subwords = 0
            for curr_token in head_mention.tokens:
                indices_inside_document = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
                for _idx in indices_inside_document:
                    idx_segment, idx_inside_segment = idx_token_to_segment[_idx]
                    curr_head_data[0].append(idx_segment)
                    curr_head_data[1].append(idx_inside_segment)
                    num_head_subwords += 1

            if num_head_subwords > self.max_span_size:
                curr_head_data[0] = curr_head_data[0][:self.max_span_size]
                curr_head_data[1] = curr_head_data[1][:self.max_span_size]
            else:
                # padding tokens index into the PAD token of the last segment
                curr_head_data[0] += [curr_head_data[0][-1]] * (self.max_span_size - num_head_subwords)
                curr_head_data[1] += [-1] * (self.max_span_size - num_head_subwords)

            head_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
            head_attention[0, num_head_subwords:] = False

            for idx_candidate, (cand_id, cand_mention) in enumerate(curr_doc.mentions.items(), start=1):
                if idx_candidate >= idx_head:
                    break

                candidates.append(cand_id)

                # Maps tokens to positions inside segments (idx_seg, idx_inside_seg) for efficient indexing later
                curr_candidate_data = [[], []]
                num_candidate_subwords = 0
                for curr_token in cand_mention.tokens:
                    indices_inside_document = mapping[(curr_token.sentence_index, curr_token.position_in_sentence)]
                    for _idx in indices_inside_document:
                        idx_segment, idx_inside_segment = idx_token_to_segment[_idx]
                        curr_candidate_data[0].append(idx_segment)
                        curr_candidate_data[1].append(idx_inside_segment)
                        num_candidate_subwords += 1

                if num_candidate_subwords > self.max_span_size:
                    curr_candidate_data[0] = curr_candidate_data[0][:self.max_span_size]
                    curr_candidate_data[1] = curr_candidate_data[1][:self.max_span_size]
                else:
                    # padding tokens index into the PAD token of the last segment
                    curr_candidate_data[0] += [curr_candidate_data[0][-1]] * (self.max_span_size - num_candidate_subwords)
                    curr_candidate_data[1] += [-1] * (self.max_span_size - num_candidate_subwords)

                candidate_data.append(curr_candidate_data)
                curr_attention = torch.ones((1, self.max_span_size), dtype=torch.bool)
                curr_attention[0, num_candidate_subwords:] = False
                candidate_attention.append(curr_attention)

                is_coreferent = cand_id in gt_antecedent_ids
                if is_coreferent:
                    correct_antecedents.append(idx_candidate)

            if len(correct_antecedents) == 0:
                correct_antecedents.append(0)

            candidate_attention = torch.cat(candidate_attention) if len(candidate_attention) > 0 else []
            all_candidate_data.append({
                "head_id": head_id,
                "head_data": torch.tensor([curr_head_data]),
                "head_attention": head_attention,
                "candidates": candidates,
                "candidate_data": torch.tensor(candidate_data),
                "candidate_attention": candidate_attention,
                "correct_antecedents": correct_antecedents
            })

        ret["preprocessed_segments"] = segments
        ret["steps"] = all_candidate_data

        return ret

    def _train_doc(self, curr_doc, eval_mode=False):
        """ Trains/evaluates (if `eval_mode` is True) model on specific document.
            Returns predictions, loss and number of examples evaluated. """

        if len(curr_doc.mentions) == 0:
            return {}, (0.0, 0)

        if not hasattr(curr_doc, "_cache_bert"):
            curr_doc._cache_bert = self._prepare_doc(curr_doc)
        cache = curr_doc._cache_bert  # type: Dict

        encoded_segments = cache["preprocessed_segments"]
        if self.freeze_pretrained:
            with torch.no_grad():
                embedded_segments = self.embedder(**{k: v.to(DEVICE) for k, v in encoded_segments.items()})
        else:
            embedded_segments = self.embedder(**{k: v.to(DEVICE) for k, v in encoded_segments.items()})

        # embedded_segments: [num_segments, max_segment_size + 3, embedding_size]
        if self.combine_layers:
            embedded_segments = torch.stack(embedded_segments["hidden_states"][-12:])
            embedded_segments, layer_weights = self.combinator(embedded_segments)
        else:
            embedded_segments = embedded_segments["last_hidden_state"]

        doc_loss, n_examples = 0.0, len(cache["steps"])
        preds = {}
        probs = {}

        for curr_step in cache["steps"]:
            head_id = curr_step["head_id"]
            head_data = curr_step["head_data"]

            candidates = curr_step["candidates"]
            candidate_data = curr_step["candidate_data"]
            correct_antecedents = curr_step["correct_antecedents"]

            # Note: num_candidates includes dummy antecedent + actual candidates
            num_candidates = len(candidates)
            if num_candidates == 1:
                curr_pred = 0
                curr_pred_prob = 1
            else:
                idx_segment = candidate_data[:, 0, :]
                idx_in_segment = candidate_data[:, 1, :]

                # [num_candidates, max_span_size, embedding_size]
                candidate_data = embedded_segments[idx_segment, idx_in_segment]
                # [1, head_size, embedding_size]
                head_data = embedded_segments[head_data[:, 0, :], head_data[:, 1, :]]
                head_data = head_data.repeat((num_candidates - 1, 1, 1))

                candidate_scores = self.scorer(candidate_data, head_data,
                                               curr_step["candidate_attention"],
                                               curr_step["head_attention"].repeat((num_candidates - 1, 1)))

                # [1, num_candidates]
                candidate_scores = torch.cat((torch.tensor([0.0], device=DEVICE),
                                              candidate_scores.flatten())).unsqueeze(0)

                candidate_probabilities = torch.softmax(candidate_scores, dim=-1)
                curr_pred_prob = torch.max(candidate_probabilities).item()

                curr_pred = torch.argmax(candidate_scores)
                doc_loss += self.loss(candidate_scores.repeat((len(correct_antecedents), 1)),
                                      torch.tensor(correct_antecedents, device=DEVICE))

            # { antecedent: [mention(s)] } pair
            existing_refs = preds.get(candidates[int(curr_pred)], [])
            existing_refs.append(head_id)
            preds[candidates[int(curr_pred)]] = existing_refs

            # { mention: probability } pair
            probs[head_id] = curr_pred_prob

        if not eval_mode:
            doc_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return preds, (float(doc_loss), n_examples), probs


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    args = parser.parse_args()
    documents = read_corpus(args.dataset)

    def create_model_instance(model_name, **override_kwargs):
        return ContextualControllerBERT(model_name=model_name,
                                        fc_hidden_size=override_kwargs.get("fc_hidden_size", args.fc_hidden_size),
                                        dropout=override_kwargs.get("dropout", args.dropout),
                                        combine_layers=override_kwargs.get("combine_layers", args.combine_layers),
                                        pretrained_model_name_or_path=override_kwargs.get("pretrained_model_name_or_path",
                                                                                          args.pretrained_model_name_or_path),
                                        learning_rate=override_kwargs.get("learning_rate", args.learning_rate),
                                        layer_learning_rate={"lr_embedder": 2e-5} if not args.freeze_pretrained else None,
                                        max_segment_size=override_kwargs.get("max_segment_size", args.max_segment_size),
                                        dataset_name=override_kwargs.get("dataset", args.dataset),
                                        freeze_pretrained=override_kwargs.get("freeze_pretrained", args.freeze_pretrained))

    # Train model
    if args.dataset == "coref149":
        INNER_K, OUTER_K = 3, 10
        logging.info(f"Performing {OUTER_K}-fold (outer) and {INNER_K}-fold (inner) CV...")
        save_path = "cache_run_contextual_bert_coref149.json"
        if args.kfold_state_cache_path is None:
            train_test_folds = KFold(n_splits=OUTER_K, shuffle=True).split(documents)
            train_test_folds = [{
                "train_docs": [documents[_i].doc_id for _i in train_dev_index],
                "test_docs": [documents[_i].doc_id for _i in test_index]
            } for train_dev_index, test_index in train_test_folds]

            fold_cache = KFoldStateCache(script_name="contextual_model_bert.py",
                                         script_args=vars(args),
                                         main_dataset=args.dataset,
                                         additional_dataset=None,
                                         fold_info=train_test_folds)
        else:
            fold_cache = KFoldStateCache.from_file(args.kfold_state_cache_path)
            OUTER_K = fold_cache.num_folds

        for curr_fold_data in fold_cache.get_next_unfinished():
            curr_train_dev_docs = list(filter(lambda doc: doc.doc_id in set(curr_fold_data["train_docs"]), documents))
            curr_test_docs = list(filter(lambda doc: doc.doc_id in set(curr_fold_data["test_docs"]), documents))
            logging.info(f"Fold#{curr_fold_data['idx_fold']}...")

            best_metric, best_name = float("inf"), None
            for idx_inner_fold, (train_index, dev_index) in enumerate(KFold(n_splits=INNER_K).split(curr_train_dev_docs)):
                curr_train_docs = [curr_train_dev_docs[_i] for _i in train_index]
                curr_dev_docs = [curr_train_dev_docs[_i] for _i in dev_index]

                curr_model = create_model_instance(model_name=f"fold{curr_fold_data['idx_fold']}_{idx_inner_fold}")
                dev_loss = curr_model.train(epochs=args.num_epochs, train_docs=curr_train_docs, dev_docs=curr_dev_docs)
                logging.info(f"Fold {curr_fold_data['idx_fold']}-{idx_inner_fold}: {dev_loss: .5f}")

                if dev_loss < best_metric:
                    best_metric = dev_loss
                    best_name = curr_model.path_model_dir

            logging.info(f"Best model: {best_name}, best loss: {best_metric: .5f}")
            curr_model = ContextualControllerBERT.from_pretrained(best_name)
            curr_test_metrics = curr_model.evaluate(curr_test_docs)
            curr_model.visualize()

            curr_test_metrics_expanded = {}
            for metric, metric_value in curr_test_metrics.items():
                curr_test_metrics_expanded[f"{metric}_p"] = float(metric_value.precision())
                curr_test_metrics_expanded[f"{metric}_r"] = float(metric_value.recall())
                curr_test_metrics_expanded[f"{metric}_f1"] = float(metric_value.f1())
            fold_cache.add_results(idx_fold=curr_fold_data["idx_fold"], results=curr_test_metrics_expanded)
            fold_cache.save(save_path)

        logging.info(f"Final scores (over {OUTER_K} folds)")
        aggregated_metrics = {}
        for curr_fold_data in fold_cache.fold_info:
            for metric, metric_value in curr_fold_data["results"].items():
                existing = aggregated_metrics.get(metric, [])
                existing.append(metric_value)

                aggregated_metrics[metric] = existing

        for metric, metric_values in aggregated_metrics.items():
            logging.info(f"- {metric}: mean={np.mean(metric_values): .4f} +- sd={np.std(metric_values): .4f}\n"
                         f"\t all fold scores: {metric_values}")
    else:
        logging.info(f"Using single train/dev/test split...")
        if args.fixed_split:
            logging.info("Using fixed dataset split")
            train_docs, dev_docs, test_docs = fixed_split(documents, args.dataset)
        else:
            train_docs, dev_docs, test_docs = split_into_sets(documents, train_prop=0.7, dev_prop=0.15, test_prop=0.15)

        model = create_model_instance(model_name=args.model_name)
        if not model.loaded_from_file:
            model.train(epochs=args.num_epochs, train_docs=train_docs, dev_docs=dev_docs)
            # Reload best checkpoint
            model = ContextualControllerBERT.from_pretrained(model.path_model_dir)

        model.evaluate(test_docs)
        model.visualize()


