from argparse import ArgumentParser
import json
import seaborn as sns


def setup_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='model file')
    parser.add_argument('-rf', '--results-files', type=str, default=None, help='json files of results')
    parser.add_argument('-pf', '--plot-folder', type=str, default=None, help='folder to save plots in')
    args = parser.parse_args()

    
def plot_accs(metrics_dict, folder):
    pass


def plot_ppls(metrics_dict, folder):
    pass


def plot_all_available(metrics_dict, folder):
    plot_accs(metrics_dict, folder)
    plot_ppls(metrics_dict, folder)


def main():
    args = setup_args()
    for results_file in args.results_files:
        with open(results_file, 'r') as f:
            results = json.loads(f.read().strip())
    plot_all_available(results, args.plot_folder)
    

if __name__ == '__main__':
    main()