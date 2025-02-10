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

"""Utility functions for setting up ReCogLab Dataset."""

from collections.abc import Sequence
import csv
import datetime
import functools
import hashlib
import importlib.resources
import json
import os
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import requests

from recoglab import common_types
from recoglab.modules import plural_nouns


_ENTITY_TYPE_TO_ENTITY_LIST_FN = {
    'plural_nouns': lambda: plural_nouns.PLURAL_NOUNS,
    'baby-names': lambda: get_baby_names()[0],
}


_T = TypeVar('_T')


def jax_sample(
    prng_key: jax.Array,
    items: Sequence[_T],
    choose_n: int,
    replace: bool = False,
) -> list[_T]:
  """Samples choose_n elements from items and returns them.

  Args:
    prng_key: jax randomization key.
    items: A list of items to draw from.
    choose_n: How many items to draw.
    replace: If true, samples with replacement.

  Returns:
    A list of items. This follow random.sample's behavior regarding mutability.
  """
  sampled_indices = jax.random.choice(
      prng_key,
      jnp.arange(len(items)),
      replace=replace,
      shape=(choose_n,),
  )
  return [items[i] for i in sampled_indices]


def jax_permutation(prng_key: jax.Array, items: Sequence[_T]) -> list[_T]:
  """Permutes the elements of items and returns a new list.

  Args:
    prng_key: jax randomization key.
    items: A list of items to permute.

  Returns:
    a permuted list.
  """
  shuffled_indices = jax.random.permutation(prng_key, jnp.arange(len(items)))
  return [items[i] for i in shuffled_indices]


def convert_split_string_to_enum(split: str) -> common_types.DatasetSplit:
  """Converts a human readable arg of string into a Enum.

  Args:
    split: string in ['train', 'val', 'test']

  Raises:
    ValueError if split isn't an acceptable value

  Returns:
    DatasetSplit Enum
  """
  return common_types.DatasetSplit(split)


def download_file(url, filename):
  """Downloads a file from a URL and saves it to the specified filename.

  Args:
      url: The URL of the file to download.
      filename: The local filename to save the downloaded file as.
  """
  try:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filename, 'wb') as file:
      for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

    print(f'File downloaded successfully to: {filename}')
  except requests.exceptions.RequestException as e:
    print(f'Error downloading file: {e}')
  except OSError as e:
    print(f'Error writing file: {e}')


def get_resource_path(resource_path):
  """Uses the relative path to get the absolute path in the package.

  Args:
      resource_path: The path to the resource.

  Returns:
      A pathlib.Path object, or None if not found.
  """
  with importlib.resources.as_file(
      importlib.resources.files('recoglab') / resource_path) as file_path:
    # file_path is now a pathlib.Path object, even if the underlying resource
    # was part of a MultiplexedPath
    return str(file_path)


BABY_NAMES_PATH = get_resource_path('data/baby-names.csv')
if not os.path.exists(BABY_NAMES_PATH):
  download_file(
      'https://raw.githubusercontent.com/hadley/data-baby-names/refs/heads/master/baby-names.csv',
      BABY_NAMES_PATH,
  )

SOCIAL_BRIDGES = {
    'friend': ' is friends with ',
    'relative': ' is related to ',
    'knows': ' knows ',
    'class': ' is in the same class as ',
    'team': ' is on the same team as ',
    'parent': ' is the parent of ',
    'sibling': ' is the sibling of ',
    'child': ' is the child of ',
    'spouse': ' is married to ',
}

SYM_SOCIAL_BRIDGES = [
    'friend', 'relative', 'knows', 'class', 'team', 'sibling', 'spouse']


@functools.lru_cache()
def load_congruent_objects(
    split: common_types.DatasetSplit = common_types.DatasetSplit.TRAIN,
) -> tuple[
    Sequence[common_types.Entity],
    tuple[Sequence[float], Sequence[float]],
    tuple[Sequence[float], Sequence[float]],
]:
  """Load congruent objects dataset from cns.

  Args:
    split: Split to load from congruent objects.

  Returns:
    A list of base entities and tuples of (lower bound list, upper bounds list)
    of size and weight respectively.
  """
  with open(get_resource_path('data/physical_object_entities.csv'), 'r') as f:
    output = [*csv.reader(f)]
    del output[-1]  # Ignore last newline row
  output = _get_split(output, split)
  entities, lb_s, ub_s, lb_w, ub_w = zip(*output[1:])
  entities = [
      common_types.Entity(entity_name, None) for entity_name in entities
  ]
  lb_s = [float(lb) for lb in lb_s]
  ub_s = [float(ub) for ub in ub_s]
  lb_w = [float(lb) for lb in lb_w]
  ub_w = [float(ub) for ub in ub_w]

  return entities, (lb_s, ub_s), (lb_w, ub_w)


@functools.lru_cache()
def get_baby_names(path: str = BABY_NAMES_PATH):
  """Load in ~200baby names + yearly stats from baby-names.csv."""
  output = []
  with open(path, 'r') as f:
    csvreader = csv.reader(f)
    for row in csvreader:
      output.append(row)

  year, name, percent_popularity, sex = zip(*output[1:])
  return name, year, percent_popularity, sex


def _get_split(
    items: Sequence[Any],
    split: common_types.DatasetSplit = common_types.DatasetSplit.TRAIN,
) -> list[Any]:
  """Get a split of the items based on the split string."""
  train_split = []
  val_split = []
  test_split = []
  for i, obj in enumerate(items):
    if i % 10 < 4:
      test_split.append(obj)
    elif i % 10 < 6:
      val_split.append(obj)
    else:
      train_split.append(obj)
  if split == common_types.DatasetSplit.TRAIN:
    return train_split
  elif split == common_types.DatasetSplit.VAL:
    return val_split
  elif split == common_types.DatasetSplit.TEST:
    return test_split
  else:
    raise ValueError(f'Unsupported split: {split}')


def get_unique_names(
    split: common_types.DatasetSplit | None = None,
) -> list[str]:
  """Get a list of random boy+girl names."""
  raw_names, *_ = get_baby_names()
  unique_arr, unique_inds = np.unique(raw_names, return_index=True)
  sorted_inds = np.argsort(unique_inds)
  names = [str(unique_arr[unique_ind]) for unique_ind in sorted_inds]
  if split:
    return _get_split(names, split)
  return names


def get_unique_entities(
    entity_type: str, split: common_types.DatasetSplit | None = None
) -> list[str]:
  """Get a list of random entities."""
  entities = _ENTITY_TYPE_TO_ENTITY_LIST_FN[entity_type]()
  entities = np.unique(entities)
  if split:
    return _get_split(entities, split)
  return entities


def get_object_names(
    split: common_types.DatasetSplit = common_types.DatasetSplit.TRAIN,
) -> list[str]:
  """Returns a list of object names based on the split.

  Args:
    split: 'train', 'val', or 'test'

  Returns:
    List of object names.
  """
  entities, *_ = load_congruent_objects(split)
  return [entity.text for entity in entities]


def stamp_config(config: ml_collections.ConfigDict, seed: int):
  config.timestamp = None
  config.config_only_hash = None
  config.seed = None
  config.config_only_hash = order_invariant_hash(config)
  stamp_creation_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  config.timestamp = stamp_creation_time
  config.seed = seed


def order_invariant_hash(config: ml_collections.ConfigDict) -> str:
  """Calculates an order-invariant hash of an ml_collections.ConfigDict.

  Args:
      config: The ConfigDict to hash.

  Returns:
      The hash value as a hexadecimal string.
  """
  items = sorted(config.items())
  json_representation = json.dumps([
      (items[0], order_invariant_hash(items[1]))
      if isinstance(items[1], ml_collections.ConfigDict)
      else (items[0], items[1])
      for items in items
  ])
  return hashlib.sha256(json_representation.encode()).hexdigest()


def load_congruent_objects_random_name(
    split: common_types.DatasetSplit = common_types.DatasetSplit.TRAIN,
    entity_name_size=5,
) -> tuple[
    Sequence[common_types.Entity],
    tuple[Sequence[float], Sequence[float]],
    tuple[Sequence[float], Sequence[float]],
]:
  """Loads entity names from a random string.

  Args:
    split: Split to load from congruent objects.
    entity_name_size: Size of the entity name.

  Returns:
    Has the same return signature as load_congruent_objects.
  """
  with open(get_resource_path('data/random_string.txt')) as f:
    random_string = f.read().strip()
  entity = []
  s = []
  w = []

  output = _get_split(range(0, 4000, entity_name_size), split)
  # random_string is 8000 characters genreated from a password generator
  for i in output:
    entity.append(random_string[i : i + entity_name_size])
    s.append(float(i) + 1)
    w.append(float(i) + 1)
  entities = [common_types.Entity(entity_name, None) for entity_name in entity]
  return entities, (s, s), (w, w)


def interleave(
    list1: list[_T], list2: list[_T], prng_key: jax.Array
) -> list[_T]:
  """Interleave two lists preserving order."""
  final_list = []
  ordering = ['list1' for _ in list1] + ['list2' for _ in list2]
  perm = jax.random.permutation(prng_key, jnp.arange(len(ordering)))
  ordering = [ordering[i] for i in perm]
  for i in ordering:
    if i == 'list1':
      final_list.append(list1.pop(0))
    elif i == 'list2':
      final_list.append(list2.pop(0))
    else:
      raise ValueError('Should be list1 or list2')
  return final_list
