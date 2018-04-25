"""TODO(jjtan): DO NOT SUBMIT without one-line documentation for run.

TODO(jjtan): DO NOT SUBMIT without a detailed description of run.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import toxicity_kaggle
import tensorflow as tf

from tensor2tensor.utils import registry
from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib

# Enable TF Eager execution
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

Modes = tf.estimator.ModeKeys
DATA_DIR = 'gs://kaggle-model-experiments/jjtan/t2t/toxicity_kaggle/data'

p = problems.problem('toxicity_kaggle')

# Get the encoders from the problem
encoders = p.feature_encoders(DATA_DIR)

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

hparams = trainer_lib.create_hparams('transformer_toxicity_kaggle', data_dir=DATA_DIR, problem_name="toxicity_kaggle")
m = registry.model('transformer_encoder')(hparams, Modes.EVAL)

def infer(inputs):
  encoded_inputs = encode(inputs)
  with tfe.restore_variables_on_create('local/output/model.ckpt-1010'):
    model_output = m.infer(encoded_inputs)["outputs"]
  return model_output

