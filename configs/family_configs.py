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

"""Configs specific to family module."""

from absl import logging
import ml_collections

from recoglab.modules import family


# Preset configs (called in presets.py)
def get_family_size_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  """Get config for Family module for family size."""
  logging.warning('Warning, there are no separate test splits for this module.')
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['family_size']
  cfg.family_size = get_family_module_config(
      relation_type=family.RelationType.FAMILY_SIZE
  )
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


def get_family_member_hobby_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  """Get config for Family module for family member hobby."""
  logging.warning('Warning, there are no separate test splits for this module.')
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['family_member_hobby']
  cfg.family_member_hobby = get_family_module_config(
      relation_type=family.RelationType.FAMILY_MEMBER_HOBBY
  )
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


def get_family_size_comparison_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  """Get config for Family module for family size comparison."""
  logging.warning('Warning, there are no separate test splits for this module.')
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['family_size_comparison']
  cfg.family_size_comparison = get_family_module_config(
      relation_type=family.RelationType.FAMILY_SIZE_COMPARISON
  )
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


def get_family_member_age_comparison_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  """Get config for Family module for family member age comparison."""
  logging.warning('Warning, there are no separate test splits for this module.')
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['family_member_age_comparison']
  cfg.family_member_age_comparison = get_family_module_config(
      relation_type=family.RelationType.FAMILY_MEMBER_AGE_COMPARISON
  )
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


def get_family_member_hobby_comparison_module_config(
    split: str = 'train',
) -> ml_collections.ConfigDict:
  """Get config for Family module for family member hobby comparison."""
  logging.warning('Warning, there are no separate test splits for this module.')
  del split
  cfg = ml_collections.ConfigDict()
  cfg.all_module_names = ['family_member_hobby_comparison']
  cfg.family_member_hobby_comparison = get_family_module_config(
      relation_type=family.RelationType.FAMILY_MEMBER_HOBBY_COMPARISON
  )
  cfg.interleave_modules = False
  cfg.combine_policy = 'none'
  return cfg


# Helper configs (called by main presets)
def get_family_module_config(
    relation_type: family.RelationType,
) -> ml_collections.ConfigDict:
  """Get config for Family module."""
  family_module_config = ml_collections.ConfigDict()
  family_module_config.name = 'FamilyModule'
  family_module_config.num_families = 50  # Sweep ~ 1 Fam = 180 tokens
  family_module_config.max_members = 5  # Sweep
  family_module_config.relation_type = relation_type
  family_module_config.preamble = ''
  family_module_config.query_preamble = ''
  family_module_config.termination_type = 'None'
  family_module_config.hop_length = -1  # -1 means sample between 1 to num_fam/2
  return family_module_config
