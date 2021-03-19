# -*- coding: utf-8 -*-


import argparse
import json
import os
import warnings

from tqdm import trange

import models


warnings.filterwarnings("ignore")


def main(args):
    prefix = "ACRE_%s_" % args.split
    template = prefix + "%06d_%02d.json"
    with open(os.path.join(args.config_folder, args.split + ".json"), "r") as f:
        gt = json.load(f)

    acc_query = 0
    acc_question = 0
    total_types = {"direct": 0, "indirect": 0, "screen_off": 0, "potential": 0}
    acc_types = {"direct": 0, "indirect": 0, "screen_off": 0, "potential": 0}
    for i in trange(len(gt)):
        label = [gt[i][j]["label"] for j in range(6, 10)]
        q_type = [gt[i][j]["type"] for j in range(6, 10)]
        data = []
        for j in range(10):
            with open(os.path.join(args.scenes_folder, template % (i, j)), "r") as f:
                data.append(json.load(f))
        model = getattr(models, args.model)(args.lower, args.upper)
        model.train(data[:6])
        pred = model.test(data[6:])
        correct = [label[k] == pred[k] for k in range(4)]
        # summarize statistics
        for l in range(4):
            total_types[q_type[l]] += 1
            if correct[l]:
                acc_types[q_type[l]] += 1
        query_correct = sum(correct)
        acc_query += query_correct
        if query_correct == 4:
            acc_question += 1
    print("Query Accuracy: {}".format(acc_query / (len(gt) * 4)))
    print("Question Accuracy: {}".format(acc_question / (len(gt))))
    print("Direct Accuracy: {}".format(acc_types["direct"] / total_types["direct"]))
    print("Indirect Accuracy: {}".format(acc_types["indirect"] / total_types["indirect"]))
    print("Screen_off Accuracy: {}".format(acc_types["screen_off"] / total_types["screen_off"]))
    print("Potential Accuracy: {}".format(acc_types["potential"] / total_types["potential"]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True,
                        help="The split to test.")
    parser.add_argument("--config_folder", type=str, required=True, 
                        help="The ground truth json file.")
    parser.add_argument("--scenes_folder", type=str, required=True,
                        help="The predicted scene json folder.")
    parser.add_argument("--model", type=str, required=True,
                        help="The model to test")
    parser.add_argument("--lower", type=float, default=0.4,
                        help="The lower bound for undetermined.")
    parser.add_argument("--upper", type=float, default=0.6,
                        help="The upper bound for undetermined.") 
    
    args = parser.parse_args()

    main(args)
