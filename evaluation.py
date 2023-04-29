import json
import os
import numpy as np


def equal_position(pos1, pos2):
    return abs(pos1[0] - pos2[0]) < 2 and abs(pos1[1] - pos2[1]) < 2


def evaluate_outputs(gt_path, ner_path, crr_path):
    with open(gt_path) as json_file:
        gt = json.load(json_file)

    with open(ner_path) as json_file:
        ner_output = json.load(json_file)

    with open(crr_path) as json_file:
        crr_output = json.load(json_file)

    ner_tp = gt.keys() & ner_output.keys()
    ner_precision = len(ner_tp) / len(ner_output)
    ner_recall = len(ner_tp) / len(gt)

    crr_tp = gt.keys() & crr_output.keys()
    crr_precision = len(crr_tp) / len(crr_output)
    crr_recall = len(crr_tp) / len(gt)

    words_tp = 0
    for key in ner_tp:
        for pos1 in gt[key]:
            if any([equal_position(pos1, pos2) for pos2 in ner_output[key]]):
                words_tp += 1

    gt_pos_len = sum([len(gt[key]) for key in ner_tp])
    ner_pos_len = sum([len(ner_output[key]) for key in ner_tp])
    # words_precision = words_tp / ner_pos_len
    # words_recall = words_tp / gt_pos_len
    words_precision = 0
    words_recall = 0

    return ner_precision, ner_recall, crr_precision, crr_recall, words_precision, words_recall


def mean_score(stories_path, ner_path, crr_path):
    scores = []
    for story in os.listdir(stories_path):
        score = np.array(evaluate_outputs(stories_path+story, ner_path+story, crr_path+story))
        scores.append(score)

    return list(zip(np.average(scores, axis=0), np.var(scores, axis=0)))


def print_output(score):
    ner_precision, ner_recall, crr_precision, crr_recall, words_precision, words_recall = tuple((round(m, 4), round(sd, 4)) for (m, sd) in score)
    print("ner precision:", ner_precision)
    print("ner recall:", ner_recall)
    print("crr precision:", crr_precision)
    print("crr recall:", crr_recall)
    print("ner words precision:", words_precision)
    print("ner words recall:", words_recall)


print_output(mean_score("data/farytales/characters/", "data/farytales/ner_output2/", "data/farytales/coreference/"))
