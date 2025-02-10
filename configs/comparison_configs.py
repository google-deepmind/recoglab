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

"""Configs specific to comparison module."""
import ml_collections

from recoglab import common_types


# Preset configs (called in presets.py)
def get_comparison_single_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['comparison']
  cfg.comparison = get_default_object_comparison_module_config()
  cfg.comparison.termination_type = 'terminal'
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


def get_comparison_people_single_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  """Generates a module for data about people's relative ages (older or younger) from which the relative age of two people can be inferred."""
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['comparison_people']
  cfg.comparison_people = get_default_comparison_module_config()
  cfg.comparison_people.termination_type = 'terminal'
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


# Helper configs (called by main presets)
def get_default_comparison_module_config() -> ml_collections.ConfigDict:
  """Get comparison module for 10 entities."""
  comparison_module_config = ml_collections.ConfigDict()
  comparison_module_config.name = 'ComparisonModule'
  comparison_module_config.num_entities_gen = 20
  comparison_module_config.ordering = common_types.Ordering.RANDOM.value
  comparison_module_config.entities_mode = common_types.EntityMode.PRESET.value
  comparison_module_config.entity_type = 'people'
  comparison_module_config.relation_type = (
      common_types.RelationType.AGE_NATURALTEXT.value
  )
  comparison_module_config.randomize_relations = True
  comparison_module_config.entities_input = None
  comparison_module_config.true_ordering = None
  comparison_module_config.do_reverse_comps = True
  comparison_module_config.preamble = ''
  comparison_module_config.query_preamble = ''
  comparison_module_config.termination_type = 'None'
  return comparison_module_config


def get_default_object_comparison_module_config() -> ml_collections.ConfigDict:
  """Get comparison module for 10 entities."""
  comparison_module_config = ml_collections.ConfigDict()
  comparison_module_config.name = 'ComparisonModule'
  comparison_module_config.num_entities_gen = 20
  comparison_module_config.ordering = common_types.Ordering.RANDOM.value
  comparison_module_config.entities_mode = common_types.EntityMode.PRESET.value
  comparison_module_config.entity_type = 'basic_objects'
  comparison_module_config.relation_type = (
      common_types.RelationType.SIZE_NATURALTEXT.value
  )
  comparison_module_config.randomize_relations = True
  comparison_module_config.entities_input = None
  comparison_module_config.true_ordering = None
  comparison_module_config.do_reverse_comps = True
  comparison_module_config.preamble = ''
  comparison_module_config.query_preamble = ''
  comparison_module_config.termination_type = 'None'
  return comparison_module_config


# Get congruent size comparisons
def get_congruent_relation_config(enum_inputs) -> ml_collections.ConfigDict:
  """Returns a config for congruent relation comparisons."""
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['comparison']
  comparison_module_config = ml_collections.ConfigDict()
  comparison_module_config.name = 'ComparisonModule'
  comparison_module_config.num_entities_gen = 20
  comparison_module_config.ordering = 'random'
  comparison_module_config.entities_mode = common_types.EntityMode.PRESET.value
  comparison_module_config.entity_type = 'congruent_objects'
  comparison_module_config.congruency_mode = enum_inputs[0]
  comparison_module_config.relation_type = enum_inputs[1]
  comparison_module_config.network_type = 'linear'
  comparison_module_config.randomize_relations = True
  comparison_module_config.preamble = ''
  comparison_module_config.query_preamble = ''
  cfg.comparison = comparison_module_config
  cfg.interleave_modules = False
  return cfg


def congruent_age_symbolic() -> ml_collections.ConfigDict:
  """Returns a config for age comparisons."""
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['comparison']
  comparison_module_config = ml_collections.ConfigDict()
  comparison_module_config.name = 'ComparisonModule'
  comparison_module_config.num_entities_gen = 20
  comparison_module_config.ordering = 'random'
  comparison_module_config.entities_mode = common_types.EntityMode.PRESET.value
  comparison_module_config.entity_type = 'people'
  comparison_module_config.relation_type = (
      common_types.RelationType.AGE_NATURALTEXT
  )
  comparison_module_config.randomize_relations = True
  comparison_module_config.preamble = ''
  comparison_module_config.query_preamble = ''
  comparison_module_config.balance_distance = True
  comparison_module_config.network_type = 'linear'
  cfg.comparison = comparison_module_config
  cfg.interleave_modules = False
  return cfg


def get_comparison_people_tree() -> ml_collections.ConfigDict:
  """Returns a config for age in a tree format."""
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['comparison']
  comparison_module_config = ml_collections.ConfigDict()
  comparison_module_config.name = 'ComparisonModule'
  comparison_module_config.num_entities_gen = 8
  comparison_module_config.ordering = 'random'
  comparison_module_config.entities_mode = common_types.EntityMode.PRESET.value
  comparison_module_config.entity_type = 'people'
  comparison_module_config.relation_type = (
      common_types.RelationType.AGE_NATURALTEXT
  )
  comparison_module_config.randomize_relations = True
  comparison_module_config.preamble = ''
  comparison_module_config.query_preamble = ''
  comparison_module_config.network_type = 'random_tree'
  cfg.comparison = comparison_module_config
  cfg.interleave_modules = False
  return cfg


def get_feasible_infeasible_people_tree() -> ml_collections.ConfigDict:
  """Returns a config for querying whether a tree problem is feasible to solve."""
  cfg = get_comparison_people_tree()
  cfg.comparison.name = 'ComparisonFeasibleModule'
  return cfg


def get_feasibility_tree() -> ml_collections.ConfigDict:
  """Returns a config for querying whether a tree problem is feasible to solve."""
  # Feasibility starts with a tree and rebalances around the answer.
  # yes, no, or unknown.
  cfg = get_comparison_people_tree()
  cfg.comparison.num_entities_gen = 10
  cfg.heuristic_rebalance_fieldname = 'answer'
  return cfg


def get_valid_people_tree() -> ml_collections.ConfigDict:
  """Returns a config for querying whether a tree problem is consistent."""
  cfg = get_comparison_people_tree()
  cfg.comparison.name = 'ComparisonValidModule'
  return cfg
