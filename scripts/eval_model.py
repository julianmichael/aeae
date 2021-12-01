from argparse import ArgumentParser
from collections import Counter
import json
import math
import os
import plot
import scipy.special
from statistics import mean

LABEL_ORDER = ['n', 'e', 'c']

def setup_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='model file')
    parser.add_argument('-p', '--predictions-file', type=str, default=None, help='file of predictor outputs')
    parser.add_argument('-tf', '--task-file', default=None, help='file containing task to compute evals on')
    parser.add_argument(
        '-mcs', '--metrics', type=str, default='all', help='metrics to calculate: \'all\' or comma separated list of any of:')
    parser.add_argument('-o', '--output-folder', type=str, default=None, help='folder to save results and plot to')  
    args = parser.parse_args()


def load_task_examples(task_file):
    examples = {}
    with open(task_file, 'r') as f:
        for l in f:
            json_dict = json.loads(l)
            examples[json_dict['uid']] = json_dict
    return examples


def load_output_instances(prediction_file):
    instances = []
    with open(prediction_file, 'r') as pf:
        for l in pf:
            instances.append(json.loads(l.strip()))
    return instances


def expected_acc(instances):
    return [
        i['human_label_counter'].get(i['model_label'], 0) /
        sum(i['human_label_counter'].values)
        for i in instances
    ]


def human_expected_acc(instances):
    return [
        i['human_label_counter'].get(i['human_label'], 0) /
        sum(i['human_label_counter'].values)
        for i in instances
    ]


def human_expected_agreement(instances):
    return [
        sum([math.pow(v / sum(i['chaos_label_counter'].values), 2) for v in i['chaos_label_counter'].values])
        for i in instances
    ]


def majority_vote_acc(instances):
    return [i['human_label'] == i['model_label'] for i in instances]


def kl_div(instances):
    return [scipy.stats.entropy(i['model_label_probs'], i['human_label_probs']) for i in instances]


def model_ppl(instances):
    return [math.exp(scipy.stats.entropy(i['model_label_probs'])) for i in instances]


def human_ppl(instances):
    return [math.exp(scipy.stats.entropy(i['human_label_probs'])) for i in instances]


METRIC_TO_FUNCTION = {
        'expected_acc': expected_acc,
        'human_expected_acc': human_expected_acc,
        'majority_vote_acc': majority_vote_acc,
        'kl_div': kl_div,
        'model_ppl': model_ppl,
        'human_ppl': human_ppl,
    }


def eval_model(task_examples, output_instances, metrics, out_folder):
    for instance in output_instances:
        uid = instance['metadata']['uid']
        task_example = task_examples.get(uid)
        instance['human_label'] = task_example['label']
        instance['human_all_labels'] = task_example['all_labels']
        instance['human_old_label'] = task_example.get('old_label')
        instance['human_old_all_labels'] = task_example.get('all_old_labels')
        instance['human_label_counter'] = task_example.get('label_counter')
        if instance['human_label_counter'] is None:
            instance['human_label_counter'] = Counter(task_example['all_labels'])
        instance['human_label_probs_list'] = [
            instance['human_label_counter'][l] / sum(instance['human_label_counter'].values()) 
            for l in LABEL_ORDER
        ]
        instance['model_label_probs_list'] = [
            instance['model_probs'][l]
            for l in LABEL_ORDER
        ]
    
    results = {}
    for metric in metrics:
        results[metric] = METRIC_TO_FUNCTION[metric](output_instances)
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
    output_instances = []
    if args.model:
        pass #run predictor
    else:
        output_instances = load_output_instances(args.prediction_file)
    metrics = args.metrics.split(',')
    if metrics[0] == 'all':
        metrics = METRIC_TO_FUNCTION.keys()
    task_examples = load_task_examples(args.task_file)

    results = eval_model(task_examples, output_instances, metrics, args.output_folder)
    if args.plot:
        plot.plot_all_available(results, args.output_folder)


if __name__ == '__main__':
    main()
