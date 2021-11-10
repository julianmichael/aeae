# Requires Python 3

from collections import namedtuple
import os.path
import urllib.request
import sys
import time
import tarfile
import zipfile

import gzip
import shutil

# copied from:
# https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
def show_progress(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

Dataset = namedtuple('Dataset', ['name', 'path', 'url', 'ext', 'description'])

# NOTE: Anthology just an example now since we don't use it for bib entries
datasets = [
    # Dataset(
    #     name = 'ACL Anthology',
    #     path = 'proposal/anthology.bib', # without `ext` (below)
    #     url = 'https://aclanthology.org/anthology.bib.gz',
    #     ext = '.gz', # or '.tar.gz' for tarballs, or '' otherwise
    #     description = "The latest ACL Anthology bibfile."
    # ),
    Dataset(
        name = 'MultiNLI',
        path = 'data/raw_data/mnli', # without `ext` (below)
        url = 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip',
        ext = '.zip',
        description = "MultiNLI dataset."
    ),
    Dataset(
        name = 'SNLI',
        path = 'data/raw_data/snli', # without `ext` (below)
        url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
        ext = '.zip',
        description = "Stanford NLI dataset."
    ),
    Dataset(
        name = 'FEVER',
        path = 'data/raw_data/nli_fever', # without `ext` (below)
        url = 'https://www.dropbox.com/s/hylbuaovqwo2zav/nli_fever.zip?dl=1',
        ext = '.zip',
        description = "FEVER (Fact Extraction and VERification) Dataset."
    ),
    # Dataset(
    #     name = 'FEVER train',
    #     path = 'data/raw_data/nli_fever/train.jsonl', # without `ext` (below)
    #     url = 'https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl',
    #     ext = '',
    #     description = "FEVER (Fact Extraction and VERification) Dataset -- train split."
    # ),
    # Dataset(
    #     name = 'FEVER dev',
    #     path = 'data/raw_data/nli_fever/dev.jsonl', # without `ext` (below)
    #     url = 'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl',
    #     ext = '',
    #     description = "FEVER (Fact Extraction and VERification) Dataset -- shared task dev split."
    # ),
    # Dataset(
    #     name = 'FEVER paper dev',
    #     path = 'data/raw_data/nli_fever/paper_dev.jsonl', # without `ext` (below)
    #     url = 'https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl',
    #     ext = '',
    #     description = "FEVER (Fact Extraction and VERification) Dataset -- paper dev split."
    # ),
    # Dataset(
    #     name = 'FEVER test',
    #     path = 'data/raw_data/nli_fever/test.jsonl', # without `ext` (below)
    #     url = 'https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl',
    #     ext = '',
    #     description = "FEVER (Fact Extraction and VERification) Dataset -- shared task test split."
    # ),
    # Dataset(
    #     name = 'FEVER paper test',
    #     path = 'data/raw_data/nli_fever/paper_test.jsonl', # without `ext` (below)
    #     url = 'https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl',
    #     ext = '',
    #     description = "FEVER (Fact Extraction and VERification) Dataset -- paper test split."
    # ),
    Dataset(
        name = 'ANLI',
        path = 'data/raw_data/anli', # without `ext` (below)
        url = 'https://dl.fbaipublicfiles.com/anli/anli_v1.0.zip',
        ext = '.zip',
        description = "Adversarial NLI dataset."
    ),
    Dataset(
        name = 'ChaosNLI',
        path = 'data/raw_data/chaosnli',
        url = 'https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1',
        ext = '.zip',
        description = "ChaosNLI dataset."
    ),
]

def get_dataset_option_prompt(num, dataset):
    if os.path.exists(dataset.path):
        color = "\u001b[32m"
        icon  = "[downloaded]"
    else:
        color = "\u001b[33m"
        icon  = "[not downloaded]"

    desc = ("\n" + dataset.description).replace("\n", "\n     ")

    return u"  {}) {}{} {}\u001b[0m ".format(num, color, dataset.name, icon) + desc + "\n"


def construct_prompt():
    prompt = "Which dataset would you like to download? ('all' to download all, 'q' to quit)\n"
    for i, dataset in enumerate(datasets):
        prompt += "\n" + get_dataset_option_prompt(i + 1, dataset)
    return prompt

def download_dataset(dataset):
    print("Downloading {}.".format(dataset.name))
    if os.path.dirname(dataset.path):
        os.makedirs(os.path.dirname(dataset.path), exist_ok=True)
    if dataset.ext in ['.tar', '.tar.gz']:
        tarpath = dataset.path + dataset.ext
        urllib.request.urlretrieve(dataset.url, tarpath, show_progress)
        result = tarfile.open(tarpath)
        result.extractall(os.path.dirname(dataset.path))
        result.close()
        os.remove(tarpath)
    elif dataset.ext == '.gz':
        gzpath = dataset.path + dataset.ext
        urllib.request.urlretrieve(dataset.url, gzpath, show_progress)
        with gzip.open(gzpath, 'rb') as f_in:
            with open(dataset.path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gzpath)
    elif dataset.ext == '.zip':
        zippath = dataset.path + dataset.ext
        urllib.request.urlretrieve(dataset.url, zippath, show_progress)
        with zipfile.ZipFile(zippath,"r") as zip_ref:
            zip_ref.extractall(dataset.path)
        os.remove(zippath)
    else:
        urllib.request.urlretrieve(dataset.url, dataset.path, show_progress)
    print("\nDownload complete: {}".format(dataset.path))

should_refresh_prompt = True
while True:
    if should_refresh_prompt:
        print(construct_prompt())
    print("Choose ({}-{}/all/q): ".format(1, len(datasets)), end='')
    should_refresh_prompt = False
    response = input()
    if "quit".startswith(response.lower()):
        break
    elif response.lower() == "all":
        for dataset in datasets:
            print(dataset.description)
            if os.path.exists(dataset.path):
                print("Already downloaded at {}.".format(dataset.path))
            else:
                download_dataset(dataset)
    else:
        try:
            dataset = datasets[int(response) - 1]
        except ValueError or IndexError:
            print("Invalid option: {}".format(response))
            continue
        if os.path.exists(dataset.path):
            print("Already downloaded at {}.".format(dataset.path))
            print("Re-download? [y/N] ", end='')
            shouldDownloadStr = input()
            if shouldDownloadStr.startswith("y") or \
               shouldDownloadStr.startswith("Y"):
                download_dataset(dataset)
                should_refresh_prompt = True
        else:
            download_dataset(dataset)
            should_refresh_prompt = True
