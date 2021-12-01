# Originally copied from: https://github.com/allenai/allennlp-models/blob/main/training_config/pair_classification/mnli_roberta.jsonnet

local transformer_model = "roberta-large";
local transformer_dim = 1024;

local data_prefix = "data/build";
local non_adversarial_datasets = ["snli", "mnli", "fever_nli"];
local adversarial_datasets = ["anli/r1", "anli/r2", "anli/r3"];
local all_datasets = non_adversarial_datasets + adversarial_datasets;
local tiny_datasets = ["tiny"];

local mode = std.extVar("MODE");
local dataset_modes = {
  tiny: tiny_datasets,
  classical: non_adversarial_datasets,
  adversarial: adversarial_datasets,
  all: all_datasets
};
local datasets = dataset_modes[mode];

local datapaths = std.map(function(p) data_prefix + "/" + p, datasets);
local split(split_name) = std.join(",", std.map(function(x) x + "/" + split_name + ".jsonl", datapaths));

{
  "dataset_reader": {
    "type": "nli",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": split("train"),
  "validation_data_path": split("dev"),
  "test_data_path": split("test"),
  "model": {
    "type": "nli_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
    },
    "dropout": 0.1,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 8
    }
  },
  "trainer": {
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "patience": 5,
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-6,
      "weight_decay": 0.1,
    }
  }
}