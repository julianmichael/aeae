from argparse import ArgumentParser
import json
import math
import os
import plot
import scipy.special
from statistics import mean

def setup_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='model file')
    parser.add_argument('-p', '--predictions-file', type=str, default=None, help='file of predictor outputs')
    parser.add_argument(
        '-mcs', '--metrics', type=str, default='all', help='metrics to calculate: \'all\' or comma separated list of any of:')
    parser.add_argument('-o', '--output-folder', type=str, default=None, help='folder to save results and plot to')  
    args = parser.parse_args()


def expected_acc(instances):
    return [
        i['chaos_label_counter'].get(i['model_label'], 0) /
        sum(i['chaos_label_counter'].values)
        for i in instances
    ]


def human_expected_acc(instances):
    return [
        i['chaos_label_counter'].get(i['chaos_majority_label'], 0) /
        sum(i['chaos_label_counter'].values)
        for i in instances
    ]


def human_expected_agreement(instances):
    return [
        sum([math.pow(v / sum(i['chaos_label_counter'].values), 2) for v in i['chaos_label_counter'].values])
        for i in instances
    ]


def majority_vote_acc(instances):
    return [i['chaos_majority_label'] == i['model_label'] for i in instances]


def kl_div(instances):
    return [scipy.stats.entropy(i['model_label_probs'], i['chaos_label_probs']) for i in instances]


def model_ppl(instances):
    return [math.exp(scipy.stats.entropy(i['model_label_probs'])) for i in instances]


def human_ppl(instances):
    return [math.exp(scipy.stats.entropy(i['chaos_label_probs'])) for i in instances]


def eval_model(instances, metrics, out_folder):

    METRIC_TO_FUNCTION = {
        'expected_acc': expected_acc,
        'human_expected_acc': human_expected_acc,
        'majority_vote_acc': majority_vote_acc,
        'kl_div': kl_div,
        'model_ppl': model_ppl,
        'human_ppl': human_ppl,
    }
    chaos_nli_examples = {} # uid -> instances dict
    for instance in instances:
        uid = instance['metadata']['uid']
        chaos_nli_instance = chaos_nli_examples.get(uid)
        if chaos_nli_instance is None:
            continue # TODO @margsli what should we do here
        instance['dataset_human_label'] = chaos_nli_instance['old_label']
        instance['dataset_all_human_labels'] = chaos_nli_instance['all_old_labels']
        instance['chaos_majority_label'] = chaos_nli_instance['label']
        instance['chaos_label_counter'] = chaos_nli_instance['label_counter']
        instance['chaos_label_probs'] = [
            chaos_nli_instance['label_counter'][l] / sum(instance['chaos_label_counter'].values) 
            for l in instance['chaos_label_counter'] # TODO @margsli: These labels need to be ordered the same as model_label_probs
        ]

        # TODO @margsli are we treating the chaos labels as gold or the original labels as gold?
    
    results = {}
    for metric in metrics:
        results[metric] = METRIC_TO_FUNCTION[metric](instances)
        print('{}: mean value {}\n'.format(metric, str(mean(results[metric]))))

    if out_folder:
        with open(os.path.join(out_folder, 'eval_results.txt')) as wf:
            for metric_name, vals in results.items():
                wf.write('{}: mean value {}\n'.format(metric_name, str(mean(vals))))
    
    return results


def main():
    args = setup_args()
    if args.model and args.prediction_file or not args.model and not args.prediction_file:
        raise RuntimeError("provide exactly one of: model file, predictions file")
    instances = []
    if args.model:
        pass #run predictor
    else:
        prediction_file = args.predictions_file
        with open(prediction_file, 'r') as pf:
            for l in pf:
                instances.append(json.loads(l.strip())) # TODO @margsli assuming jsonl format here
    metrics = args.metrics.split(',')
    if metrics[0] == 'all':
        metrics = ['expected_acc', 'majority_vote_acc', 'kl_div', 'model_ppl']
    results = eval_model(instances, metrics, args.output_folder)
    if args.plot:
        plot.plot_all_available(results, args.output_folder)


if __name__ == '__main__':
    main()
