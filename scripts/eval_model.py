from argparse import ArgumentParser
from statistics import mean


def expected_acc(instances):
    return mean([
        i['chaos_label_dist'].get(i['model_label'], 0) /
        sum(i['chaos_label_dist'].values)
        for i in instances
    ])


def majority_vote_acc(instances):
    return mean([i['label'] == i['model_label'] for i in instances])


def kl_div(instances):
    pass


def model_ppl(instances):
    pass


def eval_model(args):
    instances = None # TODO @margsli get the predictor output instances
    # TODO @margslisetup chaosnli data here
    chaos_nli_examples = {} # uid -> instances dict
    for instance in instances:
        uid = instance['metadata']['uid']
        chaos_nli_instance = chaos_nli_examples.get(uid)
        if chaos_nli_instance is None:
            continue # TODO @margsli what should we do here
        instance['dataset_human_label'] = chaos_nli_instance['old_label']
        instance['dataset_all_human_labels'] = chaos_nli_instance['all_old_labels']
        instance['chaos_majority_label'] = chaos_nli_instance['label']
        instance['chaos_label_dist'] = chaos_nli_instance['label_counter']
        # model_label = instance['model_label']
        # model_label_probs = instance['model_label_probs']

    # TODO @margsli print or store results somewhere


def setup_args():
    parser = ArgumentParser()
    # parser.add_argument('-m', '')
    # potential args to put here: 
    # 1) model file or predictions file
    # 2) original task (if we don't store all of the fields in the instance)
    # 3) which metrics to calculate
    args = parser.parse_args()

if __name__ == '__main__':
    args = setup_args()
    eval_model(args)