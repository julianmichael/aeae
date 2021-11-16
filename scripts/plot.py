import seaborn as sns
from argparse import ArgumentParser
import json

def setup_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None, help='model file')
    parser.add_argument('-rf', '--results-file', type=str, default=None, help='json file of results')
    parser.add_argument('-pf', '--plot-folder', type=str, default=None, help='folder to save plots in')
    args = parser.parse_args()

    
def plot_accs(metrics, folder):
    pass

def plot_ppls(metrics, folder):
    pass

def plot_all_available(metrics, folder):
    plot_accs(metrics, folder)
    plot_ppls(metrics, folder)

def main():
    args = setup_args()
    with open(args.results_file, 'r') as f:
        results = json.loads(f.read().strip())
    plot_all_available(results, args.plot_folder)
    

if __name__ == '__main__':
    main()