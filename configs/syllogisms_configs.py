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

"""Configs for the syllogisms module."""

from absl import logging
import ml_collections

from recoglab import common_types


# Preset configs (called in presets.py)
def get_syllogism_chain_single_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  logging.warning('Warning, there are no separate test splits for this module.')
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['syllogism_chain']
  cfg.syllogism_chain = get_syllogism_chain_config()
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


# Helper configs (called by main presets)
def get_syllogism_chain_config() -> ml_collections.ConfigDict:
  """Get config for Syllogism Chain."""
  syllogism_chain_config = ml_collections.ConfigDict()
  syllogism_chain_config.name = 'SyllogismModule'
  syllogism_chain_config.query_task = 'ValidConclusion'
  syllogism_chain_config.entities_mode = common_types.EntityMode.PRESET.value
  syllogism_chain_config.entity_type = 'plural_nouns'
  syllogism_chain_config.ordering = common_types.Ordering.RANDOM.value
  syllogism_chain_config.num_entities_max = 3
  return syllogism_chain_config
