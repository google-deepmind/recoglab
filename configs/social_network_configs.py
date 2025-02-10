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

"""Configs for the social network datasets."""

from typing import Literal

from absl import logging
import ml_collections
import numpy as np

from recoglab import common_types


# Preset configs (called in presets.py)
def get_baby_single_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  """Get config for BABY dataset.

  Args:
    split: 'train' | 'val' | 'test'

  Returns:
    Config for baby module
  """
  logging.warning('Warning, there are no separate test splits for this module.')
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['baby']
  cfg.baby = get_baby_names_module_config()
  cfg.baby.termination_type = 'terminal'
  cfg.baby.relation_type = 'friend_advanced'
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


# Helper configs (called by main presets)
def get_friends_module_config(
    mc: str, friends: list[str]
) -> ml_collections.ConfigDict:
  """Get social module config for friends."""

  friendship_module_config = ml_collections.ConfigDict()
  friendship_module_config.name = 'SocialNetworkModule'
  friendship_module_config.query_task = 'FastestMessage'
  friendship_module_config.network_type = 'random_tree'
  friendship_module_config.entities_mode = 'custom'
  friendship_module_config.entity_input = np.append(friends, mc).tolist()
  friendship_module_config.primary_entity_input = [mc]
  friendship_module_config.randomize_relations = True
  friendship_module_config.randomize_direction = False
  friendship_module_config.relation_type = 'friend'
  friendship_module_config.num_entities_max = len(friends) + 1
  friendship_module_config.erdos_renyi = dict(p=None)
  friendship_module_config.preamble = ''
  friendship_module_config.query_preamble = ''
  friendship_module_config.social_bridge_difficulty = 'basic'
  friendship_module_config.termination_type = 'None'

  return friendship_module_config


def get_family_module_config(
    mc: str, family: list[str]
) -> ml_collections.ConfigDict:
  """Get social module config for family members."""
  family_module_config = ml_collections.ConfigDict()
  family_module_config.name = 'SocialNetworkModule'
  family_module_config.query_task = 'FastestMessage'
  family_module_config.network_type = 'random_tree'
  family_module_config.entities_mode = 'custom'
  family_module_config.entity_input = np.append(family, mc).tolist()
  family_module_config.primary_entity_input = [mc]
  family_module_config.randomize_relations = True
  family_module_config.randomize_direction = False
  family_module_config.relation_type = 'relative'
  family_module_config.num_entities_max = len(family) + 1
  family_module_config.erdos_renyi = dict(p=None)
  family_module_config.preamble = ''
  family_module_config.query_preamble = ''
  family_module_config.social_bridge_difficulty = 'basic'
  family_module_config.termination_type = 'None'

  return family_module_config


def get_classmates_module_config(
    mc: str, classmates: list[str]
) -> ml_collections.ConfigDict:
  """Get social module config for classmates."""
  classmate_module_config = ml_collections.ConfigDict()
  classmate_module_config.name = 'SocialNetworkModule'
  classmate_module_config.query_task = 'FastestMessage'
  classmate_module_config.network_type = 'random_tree'
  classmate_module_config.entities_mode = 'custom'
  classmate_module_config.entity_input = np.append(classmates, mc).tolist()
  classmate_module_config.primary_entity_input = [mc]
  classmate_module_config.randomize_relations = True
  classmate_module_config.relation_type = 'class'
  classmate_module_config.randomize_direction = False
  classmate_module_config.num_entities_max = len(classmates) + 1
  classmate_module_config.erdos_renyi = dict(p=None)
  classmate_module_config.preamble = ''
  classmate_module_config.query_preamble = ''
  classmate_module_config.social_bridge_difficulty = 'basic'
  classmate_module_config.termination_type = 'None'

  return classmate_module_config


def social_tree_flower_config(
    task_name: Literal[
        'FastestMessage', 'FastestMessage_NumHops', 'FastestMessage_ExactPath'
    ],
    difficulty: str,
) -> ml_collections.ConfigDict:
  """Specifies config for flowery language relationships.

  Args:
    task_name: 'FastestMessage' | 'FastestMessage_NumHops' |
      'FastestMessage_ExactPath'
    difficulty: 'flower10' | 'flower20' | 'flower30' | 'flower40'

  Returns:
    A config.
  """
  cfg = ml_collections.ConfigDict()
  base_config = get_baby_names_module_config()
  base_config.query_task = task_name
  cfg.all_module_names = ['baby']
  if difficulty.startswith('flower'):
    base_config.relation_type = 'friend_advanced'
  else:
    base_config.relation_type = 'friend'
  if difficulty.endswith('10'):
    base_config.num_entities_max = 10
  elif difficulty.endswith('20'):
    base_config.num_entities_max = 20
  elif difficulty.endswith('30'):
    base_config.num_entities_max = 30
  elif difficulty.endswith('40'):
    base_config.num_entities_max = 40
  elif difficulty.endswith('50'): base_config.num_entities_max = 50  # pylint: disable=multiple-statements
  elif difficulty.endswith('60'): base_config.num_entities_max = 60  # pylint: disable=multiple-statements
  elif difficulty.endswith('70'): base_config.num_entities_max = 70  # pylint: disable=multiple-statements
  else:
    raise ValueError(f'Unknown difficulty size: {difficulty}')
  cfg.baby = base_config
  cfg.baby.termination_type = 'terminal'
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


def social_tree_module_config(
    task_name: Literal[
        'FastestMessage', 'FastestMessage_NumHops', 'FastestMessage_ExactPath'
    ],
    difficulty: str,
) -> ml_collections.ConfigDict:
  """Setups basic configs for social network tasks.

  Args:
    task_name: 'FastestMessage' | 'FastestMessage_NumHops' |
      'FastestMessage_ExactPath'
    difficulty: 'basic1' | 'basic2' | 'basic3' | 'hard1' | 'hard2' | 'hard3'

  Returns:
    config_dict for generating the combination of task_name and difficulty.
  """
  if 'flower' in difficulty:
    return social_tree_flower_config(task_name, difficulty)
  cfg = ml_collections.ConfigDict()
  base_config = get_baby_names_module_config()
  base_config.query_task = task_name
  cfg.all_module_names = ['baby']

  if difficulty.startswith('basic'):
    base_config.network_type = 'linear'
    base_config.relation_type = 'friend'
  elif difficulty.startswith('hard'):
    base_config.network_type = 'random_tree'
    base_config.relation_type = 'friend_advanced'
  else:
    raise ValueError(f'Unknown difficulty: {difficulty}')
  if difficulty.endswith('1'):
    base_config.num_entities_max = 10
  elif difficulty.endswith('2'):
    base_config.num_entities_max = 20
  elif difficulty.endswith('3'):
    base_config.num_entities_max = 30
  elif difficulty.endswith('4'):
    base_config.num_entities_max = 40
  else:
    raise ValueError(f'Unknown difficulty size: {difficulty}')
  cfg.baby = base_config
  cfg.baby.termination_type = 'terminal'
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


def get_baby_names_module_config() -> ml_collections.ConfigDict:
  """Get config for Social Network Module using rando baby names."""
  # Useful flags for constructing the dataset
  social_module_config = ml_collections.ConfigDict()
  social_module_config.name = 'SocialNetworkModule'
  social_module_config.network_type = 'random_tree'
  social_module_config.entities_mode = common_types.EntityMode.PRESET.value
  social_module_config.entity_type = 'baby-names'
  social_module_config.query_task = 'FastestMessage'
  social_module_config.num_entities_max = 10
  social_module_config.relation_type = 'friend'
  social_module_config.randomize_relations = True
  social_module_config.randomize_direction = True
  social_module_config.erdos_renyi = dict(p=None)
  social_module_config.preamble = ''
  social_module_config.query_preamble = (
      'Any two friends are able to pass along a message, which allows messages '
      'to move from one friend to another. Thus, messages can be passed between'
      ' two people through friends they have in common.\n')
  social_module_config.termination_type = 'None'
  return social_module_config

