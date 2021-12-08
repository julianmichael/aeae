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
    parser.add_argument('-m', '--models', type=str, default=None, help='comma-separated list of model file')
    parser.add_argument('-p', '--prediction-files', type=str, default=None, help='comma-separated list of files of predictor outputs')
    parser.add_argument('-tf', '--task-file', default=None, help='file containing task to compute evals on') # TODO @margsli generalize to multiple tasks?
    parser.add_argument(
        '-mcs', '--metrics', type=str, default='all', help='metrics to calculate: \'all\' or comma separated list of any of:')
    parser.add_argument('-o', '--output-folder', type=str, default=None, help='folder to save results and plot to')  
    return parser.parse_args()

def load_task_examples(task_file):
    examples = {}
    with open(task_file, 'r') as f:
        for l in f:
            json_dict = json.loads(l)
            examples[json_dict['uid']] = json_dict
    return examples

def load_output_instances(prediction_files_str):
    prediction_files = prediction_files_str.split(",")
    instances_dict = {}
    for prediction_file in prediction_files:
        normalized_path = os.path.normpath(prediction_file)
        path_components = normalized_path.split(os.sep)
        model_name = path_components[-3]
        task_name = path_components[-1].split('.')[0]
        instances = []
        with open(prediction_file, 'r') as pf:
            for l in pf:
                instances.append(json.loads(l.strip()))
        instances_dict[(model_name, task_name)] = instances
    return instances_dict, task_name


def expected_acc(instances):
    return [
        i['human_label_counter'].get(i['max_label'], 0) /
        sum(i['human_label_counter'].values())
        for i in instances
    ]


def human_expected_acc(instances):
    return [
        i['human_label_counter'].get(i['human_label'], 0) /
        sum(i['human_label_counter'].values())
        for i in instances
    ]


def human_expected_agreement(instances):
    return [
        sum([math.pow(v / sum(i['human_label_counter'].values()), 2) for v in i['human_label_counter'].values()])
        for i in instances
    ]


def majority_vote_acc(instances):
    return [i['human_label'] == i['max_label'] for i in instances]


def kl_div(instances):
    return [scipy.stats.entropy(i['human_label_probs'], i['model_label_probs']) for i in instances]


def model_ppl(instances):
    return [math.exp(scipy.stats.entropy(i['model_label_probs'])) for i in instances]


def human_ppl(instances):
    return [math.exp(scipy.stats.entropy(i['human_label_probs'])) for i in instances]


def human_entropy(instances):
    return [scipy.stats.entropy(i['human_label_probs']) for i in instances]


METRIC_TO_FUNCTION = {
        'expected_acc': expected_acc,
        'human_expected_acc': human_expected_acc,
        'human_expected_agreement': human_expected_agreement,
        'majority_vote_acc': majority_vote_acc,
        'kl_div': kl_div,
        'model_ppl': model_ppl,
        'human_ppl': human_ppl,
        'human_entropy': human_entropy,
    }


def eval_model(task_name, task_examples, output_instances_dict, metrics, out_folder):
    results_dict = {}

    human_instances = []
    for uid, task_example in task_examples.items():
        label_probs = {
            l: task_example['label_counter'].get(l, 0) / sum(task_example['label_counter'].values())
            for l in LABEL_ORDER
        }
        instance = {
            'uid': uid,
            'label_probs': label_probs,
            'max_label': max(list(label_probs.items()), key=lambda x: x[1])[0]
        }
        human_instances.append(instance)

    output_instances_dict[('human', task_name)] = human_instances

    for instances_name, output_instances in output_instances_dict.items():
        for instance in output_instances:
            uid = instance['uid']
            task_example = task_examples.get(uid)
            instance['human_label'] = task_example['label']
            instance['human_all_labels'] = task_example.get('all_labels')
            instance['human_old_label'] = task_example.get('old_label')
            instance['human_old_all_labels'] = task_example.get('all_old_labels')
            instance['human_label_counter'] = task_example.get('label_counter')
            if instance['human_label_counter'] is None:
                instance['human_label_counter'] = Counter(task_example['all_labels'])
            instance['human_label_probs'] = [
                instance['human_label_counter'].get(l, 0) / sum(instance['human_label_counter'].values()) 
                for l in LABEL_ORDER
            ]
            instance['model_label_probs'] = [
                instance['label_probs'][l]
                for l in LABEL_ORDER
            ]

        results = {}
        for metric in metrics:
            results[metric] = METRIC_TO_FUNCTION[metric](output_instances)
            print('{}: mean value {}\n'.format(metric, str(mean(results[metric]))))

        results_dict[instances_name] = results
        if out_folder:
            with open(os.path.join(out_folder, '{}_eval_results.txt'.format(instances_name)), mode='w') as wf:
                for metric_name, vals in results.items():
                    wf.write('{}: mean value {}\n'.format(metric_name, str(mean(vals))))
            with open(os.path.join(out_folder, '{}_eval_results.json'.format(instances_name)), mode='w') as wf:
                for metric_name, vals in results.items():
                    wf.write(json.dumps(results))

    return results_dict


def main():
    args = setup_args()
    if args.models and args.prediction_files or not args.models and not args.prediction_files:
        raise RuntimeError("provide exactly one of: model file, predictions file")
    output_instances_dict = {}
    task_name = ''
    if args.models:
        pass #run predictor
    else:
        output_instances_dict, task_name = load_output_instances(args.prediction_files)
    metrics = args.metrics.split(',')
    if metrics[0] == 'all':
        metrics = METRIC_TO_FUNCTION.keys()
    task_examples = load_task_examples(args.task_file)
    output_folder = os.path.join(args.output_folder, task_name)
    os.makedirs(output_folder, exist_ok=True)
    results_dict = eval_model(task_name, task_examples, output_instances_dict, metrics, output_folder)
    plot.plot_all_available(results_dict, output_folder)


if __name__ == '__main__':
    main()
