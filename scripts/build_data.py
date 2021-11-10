import json_utils
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import os

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_ROOT = SRC_ROOT.parent

mnli_label2std_label = defaultdict(lambda: "o")  # o stands for all other label that is invalid.
mnli_label2std_label.update({
    "entailment": "e",
    "neutral": "n",
    "contradiction": "c",
    "hidden": "h",
})

snli_label2std_label = defaultdict(lambda: "o")  # o stands for all other label that is invalid.
snli_label2std_label.update({
    "entailment": "e",
    "neutral": "n",
    "contradiction": "c",
    "hidden": "h",
})

fever_label2std_label = defaultdict(lambda: "o")
fever_label2std_label.update({
    'SUPPORTS': "e",
    'NOT ENOUGH INFO': "n",
    'REFUTES': "c",
    'hidden': "h",
})

anli_label2std_label = defaultdict(lambda: "o")
anli_label2std_label.update({
    'e': "e",
    'n': "n",
    'c': "c",
    'hidden': "h",
})

chaosnli_label2std_label = defaultdict(lambda: "o")
chaosnli_label2std_label.update({
    'e': "e",
    'n': "n",
    'c': "c",
    'hidden': "h",
})

# standard output format: {uid, premise, hypothesis, label, extra_dataset_related_field.}

def mnli2std_format(d_list, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["pairID"]
        formatted_item['premise']: str = item["sentence1"]
        formatted_item['hypothesis']: str = item["sentence2"]
        formatted_item['label']: str = mnli_label2std_label[item["gold_label"]]
        formatted_item['all_labels']: List[str] = [mnli_label2std_label[l] for l in item['annotator_labels']]
        if filter_invalid and formatted_item['label'] == 'o':
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


def snli2std_format(d_list, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["pairID"]
        formatted_item['premise']: str = item["sentence1"]
        formatted_item['hypothesis']: str = item["sentence2"]
        formatted_item['label']: str = mnli_label2std_label[item["gold_label"]]
        formatted_item['all_labels']: List[str] = [mnli_label2std_label[l] for l in item['annotator_labels']]
        if filter_invalid and formatted_item['label'] == 'o':
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


def fever_nli2std_format(d_list, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["fid"]
        formatted_item['premise']: str = item["context"]
        formatted_item['hypothesis']: str = item["query"]
        formatted_item['label']: str = fever_label2std_label[item["label"]]
        if filter_invalid and formatted_item['label'] == 'o':
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


def a_nli2std_format(d_list, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["uid"]
        formatted_item['premise']: str = item["context"]
        formatted_item['hypothesis']: str = item["hypothesis"]
        formatted_item['label']: str = anli_label2std_label[item["label"]]
        formatted_item['reason']: str = item["reason"]
        if filter_invalid and formatted_item['label'] == 'o':
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


def chaos_nli2std_format(d_list, subdataset, filter_invalid=True):
    p_list: List[Dict] = []
    for item in d_list:
        formatted_item: Dict = dict()
        formatted_item['uid']: str = item["uid"]
        formatted_item['premise']: str = item["example"]["premise"]
        formatted_item['hypothesis']: str = item["example"]["hypothesis"]
        label_dict = {}
        if subdataset == 'alphanli':
            raise NotImplementedError("no alphaNLI yet")
        elif subdataset == 'mnli_m':
            label_dict = mnli_label2std_label
        elif subdataset == 'snli':
            label_dict = snli_label2std_label
        formatted_item['old_label']: str = item["old_label"]
        formatted_item['all_old_labels']: str = [label_dict[l] for l in item.get('old_labels', [])]
        formatted_item['label_counter']: Dict = item["label_counter"]
        formatted_item['label']: str = item["majority_label"]
        if filter_invalid and (formatted_item['old_label'] == 'o' or formatted_item['label'] == 'o'):
            continue  # Skip example with invalid label.

        p_list.append(formatted_item)
    return p_list


def build_snli(path: Path):
    snli_data_root_path = (path / "snli")
    if not snli_data_root_path.exists():
        snli_data_root_path.mkdir()
    o_train = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/snli/snli_1.0/snli_1.0_train.jsonl")
    o_dev = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/snli/snli_1.0/snli_1.0_dev.jsonl")
    o_test = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/snli/snli_1.0/snli_1.0_test.jsonl")

    d_train = snli2std_format(o_train)
    d_dev = snli2std_format(o_dev)
    d_test = snli2std_format(o_test)

    print("SNLI examples without gold label have been filtered.")
    print("SNLI Train size:", len(d_train))
    print("SNLI Dev size:", len(d_dev))
    print("SNLI Test size:", len(d_test))

    json_utils.save_jsonl(d_train, snli_data_root_path / 'train.jsonl')
    json_utils.save_jsonl(d_dev, snli_data_root_path / 'dev.jsonl')
    json_utils.save_jsonl(d_test, snli_data_root_path / 'test.jsonl')


def build_mnli(path: Path):
    data_root_path = (path / "mnli")
    if not data_root_path.exists():
        data_root_path.mkdir()
    o_train = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/mnli/multinli_1.0/multinli_1.0_train.jsonl")
    o_mm_dev = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/mnli/multinli_1.0/multinli_1.0_dev_mismatched.jsonl")
    o_m_dev = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/mnli/multinli_1.0/multinli_1.0_dev_matched.jsonl")

    d_train = mnli2std_format(o_train)
    d_mm_dev = mnli2std_format(o_mm_dev)
    d_m_test = mnli2std_format(o_m_dev)

    print("MNLI examples without gold label have been filtered.")
    print("MNLI Train size:", len(d_train))
    print("MNLI MisMatched Dev size:", len(d_mm_dev))
    print("MNLI Matched dev size:", len(d_m_test))

    json_utils.save_jsonl(d_train, data_root_path / 'train.jsonl')
    json_utils.save_jsonl(d_mm_dev, data_root_path / 'mm_dev.jsonl')
    json_utils.save_jsonl(d_m_test, data_root_path / 'm_dev.jsonl')


def build_fever_nli(path: Path):
    data_root_path = (path / "fever_nli")
    if not data_root_path.exists():
        data_root_path.mkdir()

    o_train = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/nli_fever/nli_fever/train_fitems.jsonl")
    o_dev = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/nli_fever/nli_fever/dev_fitems.jsonl")
    o_test = json_utils.load_jsonl(DATA_ROOT / "data/raw_data/nli_fever/nli_fever/test_fitems.jsonl")

    d_train = fever_nli2std_format(o_train)
    d_dev = fever_nli2std_format(o_dev)
    d_test = fever_nli2std_format(o_test)

    print("FEVER-NLI Train size:", len(d_train))
    print("FEVER-NLI Dev size:", len(d_dev))
    print("FEVER-NLI Test size:", len(d_test))

    json_utils.save_jsonl(d_train, data_root_path / 'train.jsonl')
    json_utils.save_jsonl(d_dev, data_root_path / 'dev.jsonl')
    json_utils.save_jsonl(d_test, data_root_path / 'test.jsonl')


def build_anli(path: Path, round=1, version='1.0'):
    data_root_path = (path / "anli")
    if not data_root_path.exists():
        data_root_path.mkdir()

    round_tag = str(round)

    o_train = json_utils.load_jsonl(DATA_ROOT / f"data/raw_data/anli/anli_v{version}/R{round_tag}/train.jsonl")
    o_dev = json_utils.load_jsonl(DATA_ROOT / f"data/raw_data/anli/anli_v{version}/R{round_tag}/dev.jsonl")
    o_test = json_utils.load_jsonl(DATA_ROOT / f"data/raw_data/anli/anli_v{version}/R{round_tag}/test.jsonl")

    d_train = a_nli2std_format(o_train)
    d_dev = a_nli2std_format(o_dev)
    d_test = a_nli2std_format(o_test)

    print(f"ANLI (R{round_tag}) Train size:", len(d_train))
    print(f"ANLI (R{round_tag}) Dev size:", len(d_dev))
    print(f"ANLI (R{round_tag}) Test size:", len(d_test))

    if not (data_root_path / f"r{round_tag}").exists():
        (data_root_path / f"r{round_tag}").mkdir()

    json_utils.save_jsonl(d_train, data_root_path / f"r{round_tag}" / 'train.jsonl')
    json_utils.save_jsonl(d_dev, data_root_path / f"r{round_tag}" / 'dev.jsonl')
    json_utils.save_jsonl(d_test, data_root_path / f"r{round_tag}" / 'test.jsonl')


def build_chaos_nli(path: Path, version=1.0):
    data_root_path = (path / "chaosnli")
    if not data_root_path.exists():
        data_root_path.mkdir()

    for subdataset in ['alphanli', 'mnli_m', 'snli']:
        if subdataset == 'alphanli':
            print("SKIPPED ALPHANLI -- NOT IMPLEMENTED YET")
            continue
        o_sub = json_utils.load_jsonl(DATA_ROOT / f"data/raw_data/chaosnli/chaosNLI_v{version}/chaosNLI_{subdataset}.jsonl")
        d_sub = chaos_nli2std_format(o_sub, subdataset)

        print(f"Chaos {subdataset} Train size:", len(d_sub))

        if not (data_root_path / f"{subdataset}").exists():
            (data_root_path / f"{subdataset}").mkdir()

        json_utils.save_jsonl(d_sub, data_root_path / f"{subdataset}" / 'train.jsonl')


def build_data():
    processed_data_root = DATA_ROOT / "data" / "build"
    if not processed_data_root.exists():
        processed_data_root.mkdir()
    build_snli(processed_data_root)
    build_mnli(processed_data_root)
    build_fever_nli(processed_data_root)
    for round in [1, 2, 3]:
        build_anli(processed_data_root, round)
    build_chaos_nli(processed_data_root)
    print("NLI data built!")


if __name__ == '__main__':
    build_data()