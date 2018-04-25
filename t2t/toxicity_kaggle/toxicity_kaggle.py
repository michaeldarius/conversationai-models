"""TODO(jjtan): DO NOT SUBMIT without one-line documentation for model.

TODO(jjtan): DO NOT SUBMIT without a detailed description of model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

import tensorflow as tf
import pandas as pd
import os


@registry.register_problem
class ToxicityKaggle(text_problems.Text2ClassProblem):
  """Kaggle toxicity classification."""
  CLOUD_STORAGE_TRAIN = 'gs://kaggle-model-experiments/resources/train.csv'
  CLOUD_STORAGE_VALIDATION = 'gs://kaggle-model-experiments/resources/validation.csv'

  # Remove since it is automatically true because we've split the data.
  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    return ['0', '1']

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate examples."""
    del data_dir

    cloud_storage_path = {
      problem.DatasetSplit.TRAIN: self.CLOUD_STORAGE_TRAIN,
      problem.DatasetSplit.EVAL: self.CLOUD_STORAGE_VALIDATION
    }[dataset_split]

    tmp_data_file = os.path.join(tmp_dir, 'data_{}'.format(dataset_split))
    tf.gfile.Copy(cloud_storage_path, tmp_data_file, overwrite=True)

    with tf.gfile.Open(tmp_data_file, 'rb') as f:
      df = pd.read_csv(f, encoding='utf-8')
      for _, row in df.iterrows():
        yield {
          'inputs': row['comment_text'],
          'label': int(row['toxic'])
        }

  def eval_metrics(self):
    return [
       metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
       metrics.Metrics.ROC_AUC
    ]

@registry.register_hparams
def transformer_toxicity_kaggle():
 hparams = transformer.transformer_base()
 hparams.learning_rate = 3e-7
 hparams.learning_rate_constant = 3e-7
 hparams.learning_rate_warmup_steps = 0
 hparams.learning_rate_schedule = ("constant")
 return hparams
