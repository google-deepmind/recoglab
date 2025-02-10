# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Generate a tfrecord file for a ReCogLab dataset.

Usage:
python -m recoglab.generate_static_dataset \
  --recoglab_configuration_str="social_network_FastestMessage_ExactPath_flower10" \
  --split="test" \
  --output_path="/tmp/test" \
  --num_examples="5" \
  --seed="42"
"""

import os

from absl import app
from absl import flags
import tensorflow as tf

from recoglab import eval_io_lib
from recoglab import recoglab_dataset
from recoglab import utils
from recoglab.configs import presets


_CONFIG_STR = flags.DEFINE_string(
    'recoglab_configuration_str',
    'satc_single',
    'Get the preset name for the dataset we want to generate.',
)

_SPLIT = flags.DEFINE_string(
    'split',
    'train',
    'Get the split we want to generate.',
)

_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    './output/',
    'Get the output path for the dataset.',
)

_SEED = flags.DEFINE_integer(
    'seed',
    0,
    'Seed for random sampling',
)

_NUM_EXAMPLES = flags.DEFINE_integer(
    'num_examples',
    1000,
    'Get the number of examples to generate.',
)

# Config options for interleaving filler text.
_USE_FILLER = flags.DEFINE_bool(
    'use_filler',
    False,
    'Whether to use filler in the dataset. '
    'Overwrite filler parameters with _DATASET_OVERWRITE',
)

# Options for Heuristic Rejection Sampling.
_HEURISTIC_REBALANCE = flags.DEFINE_string(
    'heuristic_rebalance_fieldname',
    '',
    'If populated, will rebalance the generation around a metadata attribute.',
)

# Overwrite dataset config values
# This is provided as a way to modify a preset config from command line or
#   from a shell script.
_CONFIG_OVERWRITE = flags.DEFINE_string(
    'config_overwrite',
    '',
    'If populated, will overwrite the dataset config with this string.',
)


def overwrite_config(config, overrides_str):
  """Overwrites config parameters from a comma-separated string recursively.

  Args:
    config: The ml_collections.config_dict.ConfigDict to modify.
    overrides_str: A string of comma-separated key=value pairs.

  Returns:
    The modified ml_collections.config_dict.ConfigDict.
  """
  modify_dict = {}
  for override in overrides_str.split(','):
    key, value = override.strip().split('=')  # Handle potential whitespace
    if '.' in key:
      # modifying subconfigs
      key_token = key.split('.')
      if key_token[0] not in modify_dict:
        modify_dict[key_token[0]] = []
      modify_dict[key_token[0]].append('.'.join(key_token[1:]) + '=' + value)
    else:
      # Attempt to convert value to appropriate type (int, float, bool)
      if value.lower() == 'true':
        value = True
      elif value.lower() == 'false':
        value = False
      elif '.' in value:
        value = float(value)
      else:
        try:
          value = int(value)
        except ValueError:
          value = str(value)
      config[key] = value
  for m in modify_dict:
    overwrite_config(config[m], ','.join(modify_dict[m]))
  return config


def main(_) -> None:
  # Get the configuration string
  recoglab_configuration_str = _CONFIG_STR.value
  metadata_rebalance_field = _HEURISTIC_REBALANCE.value

  split = _SPLIT.value
  output_path = _OUTPUT_PATH.value
  num_examples = _NUM_EXAMPLES.value

  # Get dataset config and overwrite sweep values
  dataset_config = presets.get_dataset_config(
      preset_name=recoglab_configuration_str,
  )
  overwrite_values = _CONFIG_OVERWRITE.value

  # Overwrite filler config values
  if _USE_FILLER.value:
    default_filler_config = presets.default_filler_config()
    # Overwrite defaults
    for module_name in dataset_config.all_module_names:
      dataset_config[module_name].add_filler = True
      dataset_config[module_name].num_filler_lines = (
          default_filler_config.num_filler_lines
      )
      dataset_config[module_name].filler_type = (
          default_filler_config.filler_type
      )
      dataset_config[module_name].filler_position = default_filler_config.value

  if overwrite_values:
    dataset_config = overwrite_config(dataset_config, overwrite_values)
  if metadata_rebalance_field:
    dataset_config.heuristic_rebalance_fieldname = metadata_rebalance_field
  elif not hasattr(dataset_config, 'heuristic_rebalance_fieldname'):
    dataset_config.heuristic_rebalance_fieldname = ''

  # Print the dataset config for debugging
  utils.stamp_config(dataset_config, _SEED.value)
  print(dataset_config)
  examples = recoglab_dataset.generate_dataset(
      dataset_config,
      split=split,
      seed=_SEED.value,
      num_examples=num_examples,
      metadata_rebalance_field=dataset_config.heuristic_rebalance_fieldname,
  )
  save_pattern = '{config_str}_{split}'
  save_name = save_pattern.format(
      config_str=recoglab_configuration_str,
      split=split,
  )
  tfrecord = f'{save_name}.tfrecord'
  config_path = f'{save_name}.config'

  tfrecord = os.path.join(output_path, tfrecord)
  config_path = os.path.join(output_path, config_path)
  # need to do something with save_name
  index = 0
  parent_path = os.path.dirname(tfrecord)
  if not os.path.exists(parent_path):
    os.makedirs(parent_path)
  with tf.io.TFRecordWriter(tfrecord) as writer:
    assert isinstance(writer, tf.io.TFRecordWriter)
    for example in examples:
      example_proto = eval_io_lib.recoglab_dataset_example_to_tf_example(
          example, index
      )
      writer.write(example_proto.SerializeToString())
      index += 1
  with open(config_path, 'w') as f:
    f.write(dataset_config.to_json_best_effort(indent=2))


if __name__ == '__main__':
  app.run(main)
