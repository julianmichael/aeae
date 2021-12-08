from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from statistics import mean


model_order = [
    "classical", "adversarial", "all", "human"
]

def setup_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='model file')
    parser.add_argument('-rf', '--results-files', type=str, default=None, help='json files of results')
    parser.add_argument('-pf', '--plot-folder', type=str, default=None, help='folder to save plots in')
    args = parser.parse_args()
    return args

def plot_accs(metrics_dict, folder):
    bin_high = 1
    bin_low = 1/3
    num_bins = 8
    bin_ends = [round(bin_low + i / num_bins * (bin_high - bin_low), 2) for i in range(num_bins + 1)]
    def find_bin(value):
        i = 0
        while value > bin_ends[i + 1]:
            i += 1
        return i
    results_dict = {'Human Accuracy Bin': [], 'Accuracy': [], 'Model': []}

    for name, metrics in metrics_dict.items():
        human_accs = metrics.get('human_expected_acc')
        model_accs = metrics.get('expected_acc')
        binned_model_accs = {i:[] for i in range(len(bin_ends) - 1)}
        for ha, ma in zip(human_accs, model_accs):
            bin = find_bin(ha)
            binned_model_accs[bin].append(ma)
        binned_avg_model_acc = {i:mean(binned_model_accs[i]) for i in range(len(bin_ends) - 1) if len(binned_model_accs[i]) > 0}
        human_acc_bins = [i for i in range(len(bin_ends) - 1) if len(binned_model_accs[i]) > 0]
        model = [name] * len(human_acc_bins)
        if binned_avg_model_acc:
            results_dict['Model'].extend(model)
            results_dict['Human Accuracy Bin'].extend(human_acc_bins)
            results_dict['Accuracy'].extend(binned_avg_model_acc.values())

    sns.set_theme()
    df = pd.DataFrame(results_dict)
    g = sns.pointplot(x="Human Accuracy Bin", y="Accuracy", hue="Model", hue_order=model_order, data=df, ci=None, palette="muted", height=4,
            scatter_kws={"s": 50, "alpha": 1})
    labels = []
    for i, (low, high) in enumerate(zip(bin_ends, bin_ends[1:])):
        if i in results_dict['Human Accuracy Bin']:
            labels.append('[' + str(low)  + '-' + str(high) + ']')
    g.set_xticklabels(labels)
    os.makedirs(os.path.join(folder, '_'.join(metrics_dict.keys())), exist_ok=True)
    plt.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), 'acc.png'))
    plt.clf()


def plot_agreements(metrics_dict, folder):
    bin_high = 1
    bin_low = 1/3
    num_bins = 8
    bin_ends = [round(bin_low + i / num_bins * (bin_high - bin_low), 2) for i in range(num_bins + 1)]
    def find_bin(value):
        i = 0
        while value > bin_ends[i + 1]:
            i += 1
        return i
    results_dict = {'Human Agreement Bin': [], 'Accuracy': [], 'Model': []}

    for name, metrics in metrics_dict.items():
        human_agreement = metrics.get('human_expected_agreement')
        model_agreement = metrics.get('expected_acc')
        binned_model_agreement = {i:[] for i in range(len(bin_ends) - 1)}
        for ha, ma in zip(human_agreement, model_agreement):
            bin = find_bin(ha)
            binned_model_agreement[bin].append(ma)
        binned_avg_model_acc = {i:mean(binned_model_agreement[i]) for i in range(len(bin_ends) - 1) if len(binned_model_agreement[i]) > 0}
        human_agg_bins = [i for i in range(len(bin_ends) - 1) if len(binned_model_agreement[i]) > 0]
        model = [name] * len(human_agg_bins)
        if binned_avg_model_acc:
            results_dict['Model'].extend(model)
            results_dict['Human Agreement Bin'].extend(human_agg_bins)
            results_dict['Accuracy'].extend(binned_avg_model_acc.values())

    sns.set_theme()
    df = pd.DataFrame(results_dict)
    g = sns.pointplot(x="Human Agreement Bin", y="Accuracy", hue="Model", data=df, ci=None, palette="muted", height=4,
            scatter_kws={"s": 20, "alpha": 0.5})
    labels = []
    for i, (low, high) in enumerate(zip(bin_ends, bin_ends[1:])):
        if i in results_dict['Human Agreement Bin']:
            labels.append('[' + str(low)  + '-' + str(high) + ']')
    g.set_xticklabels(labels)
    os.makedirs(os.path.join(folder, '_'.join(metrics_dict.keys())), exist_ok=True)
    plt.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), 'agreement.png'))
    plt.clf()


from math import exp

def plot_kldivs(metrics_dict, folder, exponentiate=False):
    results_dict = {'kl_div': [], 'human_entropy': [], 'model': []}
    for name, metrics in metrics_dict.items():
        human_entropy = metrics.get('human_entropy')
        kl_div = metrics.get('kl_div')
        model = [name] * len(human_entropy)
        if human_entropy:
            hent = map(exp, human_entropy) if exponentiate else human_entropy
            results_dict['human_entropy'].extend(hent)
            kl = map(exp, kl_div) if exponentiate else kl_div
            results_dict['kl_div'].extend(kl)
            results_dict['model'].extend(model)
    sns.set_theme()
    df = pd.DataFrame(results_dict)
    g = sns.lmplot(
        data=df,
        x="human_entropy", y="kl_div", hue="model",
        height=5,
        scatter_kws={"s": 5, "alpha": 0.3}
    )
    y_label = "Human Perplexity" if exponentiate else "Human Entropy"
    x_label = "Exp(KL)" if exponentiate else "KL-Divergence"
    g.set_axis_labels(y_label, x_label)
    filename = "kldiv_exp.png" if exponentiate else "kldiv.png"
    os.makedirs(os.path.join(folder, '_'.join(metrics_dict.keys())), exist_ok=True)
    plt.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), filename))
    plt.clf()

def plot_ppls(metrics_dict, folder):
    results_dict = {'human_ppl': [], 'model_ppl': [], 'model': []}
    for name, metrics in metrics_dict.items():
        human_ppl = metrics.get('human_ppl')
        model_ppl = metrics.get('model_ppl')
        model = [name] * len(model_ppl)
        if model_ppl and human_ppl:
            results_dict['human_ppl'].extend(human_ppl)
            results_dict['model_ppl'].extend(model_ppl)
            results_dict['model'].extend(model)
    sns.set_theme()
    df = pd.DataFrame(results_dict)
    g = sns.lmplot(
        data=df,
        x="human_ppl", y="model_ppl", hue="model",
        height=5,
        scatter_kws={"s": 5, "alpha": 0.3}
    )
    g.set_axis_labels("Human Perplexity", "Model Perplexity")
    os.makedirs(os.path.join(folder, '_'.join(metrics_dict.keys())), exist_ok=True)
    plt.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), 'ppl.png'))
    plt.clf()

def plot_ppls_joint(metrics_dict, folder):
    results_dict = {'human_ppl': [], 'model_ppl': [], 'model': []}
    for name, metrics in metrics_dict.items():
        human_ppl = metrics.get('human_ppl')
        model_ppl = metrics.get('model_ppl')
        model = [name] * len(model_ppl)
        if model_ppl and human_ppl:
            results_dict['human_ppl'].extend(human_ppl)
            results_dict['model_ppl'].extend(model_ppl)
            results_dict['model'].extend(model)
    sns.set_theme()
    df = pd.DataFrame(results_dict)

    g = sns.jointplot(
        data=df,
        x="human_ppl", y="model_ppl",
        hue="model",
        hue_order=model_order,
        xlim=(0.99, 3.01),
        ylim=(0.99, 3.01),
        height=5,
        joint_kws={"s": 4, "alpha": 0.7},
        marginal_kws={"clip": [1.0, 3.0]}
    )
    for _, gr in sorted(list(df.groupby("model")), key = lambda x: model_order.index(x[0])):
        sns.regplot(
            data = gr,
            x="human_ppl", y="model_ppl",
            # xlim=(0.9, 3.1),
            # ylim=(0.9, 3.1),
            # height=5,
            scatter=False,
            ax=g.ax_joint,
            truncate=False
        )
    # g.plot_joint(sns.regplot)
    g.set_axis_labels("Human Perplexity", "Model Perplexity")
    os.makedirs(os.path.join(folder, '_'.join(metrics_dict.keys())), exist_ok=True)
    plt.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), 'ppl_joint.png'))
    plt.clf()


def plot_all_available(metrics_dict, folder):
    plot_accs(metrics_dict, folder)
    plot_agreements(metrics_dict, folder)
    plot_kldivs(metrics_dict, folder, exponentiate = False)
    plot_kldivs(metrics_dict, folder, exponentiate = True)
    plot_ppls(metrics_dict, folder)
    plot_ppls_joint(metrics_dict, folder)


def main():
    args = setup_args()
    results = {}
    task_name = ''
    for results_file in args.results_files.split(','):
        normalized_path = os.path.normpath(results_file)
        path_components = normalized_path.split(os.sep)
        model_name = path_components[-3]
        task_name = path_components[-1].split('.')[0]
        with open(results_file, 'r') as f:
            results[model_name] = json.loads(f.read().strip())
    plot_task_folder = os.path.join(args.plot_folder, task_name)
    os.makedirs(plot_task_folder, exist_ok=True)
    plot_all_available(results, plot_task_folder)
    

if __name__ == '__main__':
    main()
