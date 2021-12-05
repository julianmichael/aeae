from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from statistics import mean


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
            results_dict['Accuracy'].extend(binned_avg_model_acc)

    sns.set_theme()
    df = pd.DataFrame(results_dict)
    g = sns.pointplot(x="Human Accuracy Bin", y="Accuracy", hue="Model", data=df, ci=None, palette="muted", height=4,
            scatter_kws={"s": 50, "alpha": 1})
    labels = []
    for i, (low, high) in enumerate(zip(bin_ends, bin_ends[1:])):
        if i in results_dict['Human Accuracy Bin']:
            labels.append('[' + str(low)  + '-' + str(high) + ']')
    g.set_xticklabels(labels)
    os.makedirs(os.path.join(folder, '_'.join(metrics_dict.keys())), exist_ok=True)
    plt.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), 'acc.png'))


# def plot_agreement(metrics_dict, folder):
#     bin_ends = [.33, .5, .67, .83, 1]

#     def find_bin(value):
#         i = 0
#         while value < bin_ends[i]:
#             i += 1
#         return i
#     results_dict = {'Human Accuracy Bin': None, 'Accuracy': [], 'Model': []}

#     for name, metrics in metrics_dict.item():
#         human_accs = metrics.get('human_expected_acc')
#         model_accs = metrics.get('expected_acc')
#         binned_model_accs = {i:[] for i in range(len(bin_ends) - 1)]}
#         human_acc_bins = [i in range(len(bin_ends) - 1)]
#         for ha, ma in zip(human_accs, model_accs):
#             bin = find_bin(ha)
#             binned_model_accs[bin].append(ma)
#         binned_avg_model_acc = {i:mean(binned_model_accs[i]) for i in range(len(bin_ends) - 1)]}
#         model = [name] * len(human_acc_bins)
#         if binned_avg_model_acc:
#             results_dict['Model'].extend(model)
#             results_dict['Human Accuracy Bin'].extend(human_acc_bins)
#             results_dict['Accuracy'].extend(binned_avg_model_acc)

#     sns.set_theme()
#     df = pd.DataFrame(results_dict)
#     g = sns.pointplot(x="Human Accuracy Bin", y="Accuracy", hue="Model", data=df, ci=None, palette="muted", height=4,
#             scatter_kws={"s": 50, "alpha": 1})
#     g.set_xticklabels(['[0.33-0.45]','[0.45-0.6]','[0.6-0.7]','[0.7-0.85]','[0.85-1.0]'])
#     fig = g.get_fig()
#     fig.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), 'acc.png'))



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
        height=5
    )
    g.set_axis_labels("Human Perplexity", "Model Perplexity")
    os.makedirs(os.path.join(folder, '_'.join(metrics_dict.keys())), exist_ok=True)
    plt.savefig(os.path.join(folder, '_'.join(metrics_dict.keys()), 'ppl.png'))


def plot_all_available(metrics_dict, folder):
    plot_accs(metrics_dict, folder)
    plot_ppls(metrics_dict, folder)


def main():
    args = setup_args()
    results = {}
    for results_file in args.results_files.split(','):
        filename = os.path.basename(results_file).split('.')[0]
        with open(results_file, 'r') as f:
            results[filename] = json.loads(f.read().strip())
    plot_all_available(results, args.plot_folder)
    

if __name__ == '__main__':
    main()