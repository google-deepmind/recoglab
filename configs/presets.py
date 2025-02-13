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

"""Configs for the datasets."""

from collections.abc import Mapping
from typing import Optional

import immutabledict
import ml_collections

from recoglab import common_types
from recoglab.configs import comparison_configs
from recoglab.configs import family_configs
from recoglab.configs import social_network_configs
from recoglab.configs import syllogisms_configs


def default_filler_config() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict({
      'num_filler_lines': 0,
      # random_text | entity_filler | extra_modules
      'filler_type': 'random_text',
      # before | after | interspersed
      'filler_position': 'interspersed',
  })


def generate_social_network_iclr_configs() -> (
    Mapping[str, ml_collections.ConfigDict]
):
  """Returns a mapping to ICLR configs on Social Network tasks."""
  difficulty_levels = (
      # for seeding experiments
      'basic1',
      'basic2',
      'basic3',
      'hard1',
      'hard2',
      'hard3',
      'hard4',
      'basic4',
      # for flower experiments
      'flower10',
      'flower20',
      'flower30',
      'flower40',
      'flower50',
      'flower60',
      'flower70',
      'noflower10',
      'noflower20',
      'noflower30',
      'noflower40',
      'noflower50',
      'noflower60',
      'noflower70',
  )
  fastest_message_tasks = (
      'FastestMessage_NumHops', 'FastestMessage_ExactPath')

  iclr_configs = {}

  for difficulty in difficulty_levels:
    for task in fastest_message_tasks:
      iclr_configs[f'social_network_{task}_{difficulty}'] = (
          social_network_configs.social_tree_module_config(task, difficulty)
      )
  return immutabledict.immutabledict(iclr_configs)


def generate_comparison_iclr_configs() -> (
    Mapping[str, ml_collections.ConfigDict]
):
  """Returns a mapping to ICLR configs on Comparison tasks."""
  premade_comparison_configs = {
      'comparison_congruent_size': (
          comparison_configs.get_congruent_relation_config((
              common_types.CongruencyMode.ALL_CONGRUENT,
              common_types.RelationType.SIZE_NATURALTEXT,
          ))
      ),
      'comparison_incongruent_size': (
          comparison_configs.get_congruent_relation_config((
              common_types.CongruencyMode.ALL_INCONGRUENT,
              common_types.RelationType.SIZE_NATURALTEXT,
          ))
      ),
      'comparison_congruent_weight': (
          comparison_configs.get_congruent_relation_config((
              common_types.CongruencyMode.ALL_CONGRUENT,
              common_types.RelationType.WEIGHT_NATURALTEXT,
          ))
      ),
      'comparison_incongruent_weight': (
          comparison_configs.get_congruent_relation_config((
              common_types.CongruencyMode.ALL_INCONGRUENT,
              common_types.RelationType.WEIGHT_NATURALTEXT,
          ))
      ),
      'comparison_random_string_weight': (
          comparison_configs.get_congruent_relation_config((
              common_types.CongruencyMode.RANDOM_NAME,
              common_types.RelationType.WEIGHT_NATURALTEXT,
          ))
      ),
      'comparison_random_string_size': (
          comparison_configs.get_congruent_relation_config((
              common_types.CongruencyMode.RANDOM_NAME,
              common_types.RelationType.SIZE_NATURALTEXT,
          ))
      ),
      'feasible_infeasible_tree': (
          comparison_configs.get_comparison_people_tree()
      ),
      'consistent_inconsistent_tree': (
          comparison_configs.get_valid_people_tree()
      ),
      'symoblic_distance_age': comparison_configs.symoblic_distance_age(),
  }
  return immutabledict.immutabledict(premade_comparison_configs)


ICLR_CONFIGS = {
    **generate_social_network_iclr_configs(),
    **generate_comparison_iclr_configs(),
}


PRESET_CONFIGS = {
    # Social Network Presets
    'baby_single': social_network_configs.get_baby_single_module_config,

    # Comparison Presets
    'comparison_single': comparison_configs.get_comparison_single_module_config,
    'comparison_people': (
        comparison_configs.get_comparison_people_single_module_config
    ),

    # Family JSON Presets
    'family_size': family_configs.get_family_size_module_config,
    'family_member_hobby': family_configs.get_family_member_hobby_module_config,
    'family_size_comparison': (
        family_configs.get_family_size_comparison_module_config
    ),
    'family_member_age_comparison': (
        family_configs.get_family_member_age_comparison_module_config
    ),
    'family_member_hobby_comparison': (
        family_configs.get_family_member_hobby_comparison_module_config
    ),

    # Syllogisms Presets
    'syllogism_chain': (
        syllogisms_configs.get_syllogism_chain_single_module_config
    ),
}


# Outside function to call to get the dataset config.
def get_dataset_config(
    preset_name: str = 'comparison_single',
    input_config: Optional[ml_collections.ConfigDict] = None,
) -> ml_collections.ConfigDict:
  """Get config for ReCogLab dataset.

  Args:
    preset_name: 'satc_single' | 'comparison_single', specify config or use a
      default loadout (default='comparison_single')
    input_config: config to pass in if using 'custom' preset

  Returns:

  Returns:
    config to feed into recoglab_dataset.ReCogLabDatasetGenerator().
  """
  # Useful flags for constructing the dataset
  # If passing in a preset name, get the config
  if preset_name in PRESET_CONFIGS:
    cfg = PRESET_CONFIGS[preset_name]('unused')
  elif preset_name in ICLR_CONFIGS:
    cfg = ICLR_CONFIGS[preset_name]
  else:
    raise ValueError(
        f'Preset {preset_name} not found in PRESET_CONFIGS or ICLR_CONFIGS.'
    )

  # Set filler config values
  if input_config and input_config.add_filler:
    for module_name in cfg.all_module_names:
      cfg[module_name].add_filler = True
      cfg[module_name].num_filler_lines = input_config[
          module_name
      ].num_filler_lines
      cfg[module_name].filler_type = input_config[module_name].filler_type
      cfg[module_name].filler_position = input_config[
          module_name
      ].filler_position
  else:
    for module_name in cfg.all_module_names:
      cfg[module_name].add_filler = False
      cfg[module_name].num_filler_lines = 0
      cfg[module_name].filler_type = ''
      cfg[module_name].filler_position = ''

  # Set some more config values
  if input_config and 'query_preamble' in input_config.keys():
    # Can be overridden by module specific preamble if provided, but in
    # multi-module setups, the prompt may require context from all the modules.
    cfg.query_preamble = input_config.query_preamble
  else:
    cfg.query_preamble = ''
  cfg.use_images = False
  cfg.block_sep_text = '\n'
  if input_config and 'maintain_entity_uniqueness' in input_config.keys():
    cfg.maintain_entity_uniqueness = input_config.maintain_entity_uniqueness
  else:
    cfg.maintain_entity_uniqueness = False

  return cfg
