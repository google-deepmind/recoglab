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

"""Code for generating Comparison ReCogLab dataset."""

import copy
import functools
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import ml_collections
import networkx as nx
import numpy as np

from recoglab import common_types
from recoglab import utils
from recoglab.modules import recoglab_base


CongruencyMode = common_types.CongruencyMode
RelationType = common_types.RelationType


def attribute_sort_key(
    item: common_types.CongruentObjectEntity, attribute: RelationType
) -> float | int:
  match attribute:
    case RelationType.SIZE_NATURALTEXT:
      return item.size
    case RelationType.WEIGHT_NATURALTEXT:
      return item.weight
    case _:
      raise NotImplementedError('Unable to generate entities')


class GenerateCongruentEntities:
  """A generator to produce entities that obey congruency mode."""

  def __init__(
      self,
      congruency_mode: CongruencyMode,
      split: common_types.DatasetSplit = common_types.DatasetSplit.TRAIN,
  ):
    """Initializes a generator to produce entities that obey Congruency rules.

    Args:
      congruency_mode: determines how congruent the entity sets will be.
      split: Unused, but will dictate which entities are available.
    """
    self._congruency_mode = congruency_mode
    if self._congruency_mode == CongruencyMode.RANDOM_NAME:
      (self._objects, self._size_bounds, self._weight_bounds) = (
          utils.load_congruent_objects_random_name(split)
      )
    else:
      (self._objects, self._size_bounds, self._weight_bounds) = (
          utils.load_congruent_objects(split)
      )
    self._lower_bound = [
        common_types.CongruentObjectEntity(e.text, e.image, lb_s, lb_w)
        for e, lb_s, lb_w in zip(
            self._objects, self._size_bounds[0], self._weight_bounds[0]
        )
    ]
    self._upper_bound = [
        common_types.CongruentObjectEntity(e.text, e.image, ub_s, ub_w)
        for e, ub_s, ub_w in zip(
            self._objects, self._size_bounds[1], self._weight_bounds[1]
        )
    ]
    self._n_objects = len(self._objects)

  def generate_set_of_entities(
      self, prng_key: jax.Array, n_entities: int, attribute: RelationType
  ) -> list[common_types.CongruentObjectEntity]:
    """Selects n_entities from the congruency objects and orders them.

    Args:
      prng_key: the PRNG key used for this example
      n_entities: The number of entities to draw
      attribute: Which attribute to sort the examples by

    Returns:
      A list of entities that are sorted according to the CongruencyMode passed
      at initialization.
    """
    prng_key, subkey = jax.random.split(prng_key)
    selected_indices = utils.jax_sample(
        subkey, range(self._n_objects), n_entities, replace=False
    )
    prng_key, subkey1, subkey2 = jax.random.split(prng_key, 3)
    size_random = jax.random.uniform(subkey1, shape=[n_entities])
    weight_random = jax.random.uniform(subkey2, shape=[n_entities])
    lower = jnp.array([self._lower_bound[i].size for i in selected_indices])
    upper = jnp.array([self._upper_bound[i].size for i in selected_indices])
    sampled_size = lower + (upper - lower) * size_random

    lower = jnp.array([self._lower_bound[i].weight for i in selected_indices])
    upper = jnp.array([self._upper_bound[i].weight for i in selected_indices])
    sampled_weight = lower + (upper - lower) * weight_random
    sampled_obj = [
        common_types.CongruentObjectEntity(
            self._objects[i].text, self._objects[i].image, float(ss), float(sw)
        )
        for i, ss, sw in zip(selected_indices, sampled_size, sampled_weight)
    ]
    if attribute != RelationType.AGE_NATURALTEXT:
      sampled_obj.sort(
          key=functools.partial(attribute_sort_key, attribute=attribute)
      )
    # sampled_obj this will be ascending.
    match self._congruency_mode:
      case CongruencyMode.ALL_CONGRUENT:
        pass
      case CongruencyMode.ALL_INCONGRUENT:
        # Invert to make all relations incongruent.
        sampled_obj = sampled_obj[::-1]
      case CongruencyMode.RANDOM:
        prng_key, subkey = jax.random.split(prng_key)
        sampled_obj = utils.jax_permutation(subkey, sampled_obj)
      case CongruencyMode.RANDOM_NAME:
        prng_key, subkey = jax.random.split(prng_key)
        sampled_obj = utils.jax_permutation(subkey, sampled_obj)
    return sampled_obj


class ComparisonGenerator:
  """Generate setences that describe a particular relationship."""

  def __init__(self, cfg: ml_collections.ConfigDict):
    # this should prob be args or cfg.
    self.cfg = cfg

  def generate(
      self,
      lesser_entity: common_types.Entity,
      greater_entity: common_types.Entity,
      prng_key: jax.Array,
  ) -> recoglab_base.ReCogLabBlock:
    """Generates a block describing lesser_entity and greater_entity.

    Args:
      lesser_entity: An entity that is less than greater_entity
      greater_entity: An entity that is greater than lesser_entity
      prng_key: Jax random key

    Returns:
      ReCogLabBlock containing relevant information.
    """
    block = recoglab_base.ReCogLabBlock()
    _, subkey = jax.random.split(prng_key)
    flip_comparison = float(jax.random.uniform(subkey, shape=())) < 0.5
    if flip_comparison:
      lesser_entity, greater_entity = greater_entity, lesser_entity
    match self.cfg.relation_type:
      case RelationType.SIZE_NATURALTEXT.value:
        if flip_comparison:
          relation_separator = ' is larger than '
        else:
          relation_separator = ' is smaller than '
      case RelationType.AGE_NATURALTEXT.value:
        if flip_comparison:
          relation_separator = ' is older than '
        else:
          relation_separator = ' is younger than '
      case RelationType.WEIGHT_NATURALTEXT.value:
        if flip_comparison:
          relation_separator = ' is heavier than '
        else:
          relation_separator = ' is lighter than '
      case _:
        raise NotImplementedError('Need to implement this relation type')
    block.prompt.append(
        f'{lesser_entity.text}{relation_separator}{greater_entity.text}'
    )
    return block


def generate_comparison_query_block(
    cfg: ml_collections.ConfigDict,
    entities: list[common_types.Entity],
    prng_key: jax.Array,
    graph,
) -> recoglab_base.ReCogLabQuestionAnswerBlock:
  """Generate a query and answer.

  Args:
    cfg: ConfigDict to control the generation of query block
    entities: List of entities to generate query from
    prng_key: jax random key.
    graph: the graph structure if a configuration requires a grpah to construct
      the query

  Returns:
    A question comparison to evaluate about a set of entities.
  """
  block = recoglab_base.ReCogLabQuestionAnswerBlock()
  # Decide on the comparison
  prng_key, subkey = jax.random.split(prng_key)
  sampled_indices = jax.random.choice(
      subkey,
      jnp.arange(len(entities)),
      replace=False,
      shape=(2,),
  )
  if cfg.network_type == 'linear':
    if cfg.get('balance_distance'):
      # ignore sampled indices and generate new ones
      prng_key, subkey1, subkey2, subkey3 = jax.random.split(prng_key, 4)
      distance_ = int(
          jax.random.randint(subkey1, minval=1, maxval=len(entities), shape=())
      )
      lower_ind = int(
          jax.random.randint(
              subkey2, minval=0, maxval=len(entities) - distance_, shape=()
          )
      )
      if float(jax.random.uniform(subkey3, shape=())) < 0.5:
        sampled_indices = [lower_ind, lower_ind + distance_]
      else:
        sampled_indices = [lower_ind + distance_, lower_ind]
    query_entities = [entities[i] for i in sampled_indices]
    index_q_ent1, index_q_ent2 = sampled_indices
    distance = abs(index_q_ent1 - index_q_ent2)
    if index_q_ent1 < index_q_ent2:
      query_less_ent, query_greater_ent = query_entities
    elif index_q_ent1 > index_q_ent2:
      query_greater_ent, query_less_ent = query_entities
    else:
      raise ValueError('The indices should never be equal')
  elif cfg.network_type == 'random_tree':
    query_entities = [entities[i] for i in sampled_indices]
    index_q_ent1, index_q_ent2 = sampled_indices
    index_q_ent1 = int(index_q_ent1)
    index_q_ent2 = int(index_q_ent2)
    # check the graph now.
    q1_to_q2_exists = nx.has_path(graph, index_q_ent1, index_q_ent2)
    q2_to_q1_exists = nx.has_path(graph, index_q_ent2, index_q_ent1)
    if q1_to_q2_exists:
      distance = nx.shortest_path_length(graph, index_q_ent1, index_q_ent2)
      query_less_ent, query_greater_ent = query_entities
    elif q2_to_q1_exists:
      distance = nx.shortest_path_length(graph, index_q_ent2, index_q_ent1)
      query_greater_ent, query_less_ent = query_entities
    else:
      distance = -1
      query_less_ent, query_greater_ent = query_entities
  else:
    raise NotImplementedError(
        f'Need to implement this network type: {cfg.network_type}'
    )
  query_greater = query_greater_ent.text
  query_less = query_less_ent.text

  # Generate the query and answer
  block.prompt.append(cfg.query_preamble)
  match cfg.relation_type:
    case RelationType.SIZE_NATURALTEXT:
      pattern = 'Is %s %s %s?'
      lesser_clause = 'smaller than'
      greater_clause = 'larger than'
    case RelationType.AGE_NATURALTEXT:
      pattern = 'Is %s %s %s?'
      lesser_clause = 'younger than'
      greater_clause = 'older than'
    case RelationType.WEIGHT_NATURALTEXT:
      pattern = 'Is %s %s %s?'
      lesser_clause = 'lighter than'
      greater_clause = 'heavier than'
    case _:
      raise NotImplementedError('Need to implement this relation type')
  prng_key, subkey = jax.random.split(prng_key)
  rand1, rand2 = jax.random.uniform(key=subkey, shape=[2])
  if float(rand1) < 0.5:
    clause = lesser_clause
    answer = False
  else:
    clause = greater_clause
    answer = True
  if float(rand2) < 0.5:
    # Flip the entities
    answer = not answer
    entity1, entity2 = query_less, query_greater
  else:
    entity1, entity2 = query_greater, query_less
  if distance == -1:
    # No path exists so the answer is unknown.
    block.answers = ['Unknown']
  elif answer:
    block.answers = ['Yes']
  else:
    block.answers = ['No']
  block.prompt.append(pattern % (entity1, clause, entity2))
  block.metadata['distance'] = str(distance)
  block.metadata['entities'] = repr([e.text for e in entities])
  # To balance and slice by ansewr.
  block.metadata['answer'] = block.answers[0]
  return block


class ComparisonModule(recoglab_base.ReCogLabModule):
  """Data class for comparison training example.

  distance: int metadata for how far apart the comparisons are in the chain.
  """

  def get_metadata(self) -> Dict[str, str]:
    """Returns the metadata for a given module."""
    if self.answer_block:
      return {
          'distance': self.answer_block.metadata['distance'],
          'entities': self.answer_block.metadata['entities'],
          'answer': self.answer_block.answers[0],
      }
    return {}


class ComparisonCoherentEvidenceModule(recoglab_base.ReCogLabModule):
  """Data class for evaluating whether all comparisons are coherent.

  distance: int metadata for how far apart the comparisons are in the chain.
  """

  def get_metadata(self) -> Dict[str, str]:
    """Returns the metadata for a given module."""
    if self.answer_block:
      return {'answer': self.answer_block.answers[0],
              'entities': self.answer_block.metadata['entities']}
    return {}


class ComparisonModuleGenerator:
  """Generates a module of comparison blocks."""

  def __init__(
      self,
      cfg: ml_collections.ConfigDict,
      jax_init_key: jax.Array,
      split: common_types.DatasetSplit,
  ):
    super().__init__()
    self.cfg = cfg
    self.split = split

    del jax_init_key
    self.all_entities = []
    self.comparison_generator = ComparisonGenerator(cfg)

    # Set the entities
    if self.cfg.entities_mode == common_types.EntityMode.PRESET.value:
      if self.cfg.entity_type == 'congruent_objects':
        self.congruency_generator = GenerateCongruentEntities(
            self.cfg.congruency_mode, self.split
        )
      elif self.cfg.entity_type == 'basic_objects':
        # Access physical_entities
        self.all_entities = utils.get_object_names(split)
        self.all_entities = [
            common_types.Entity(entity, None) for entity in self.all_entities
        ]
      elif self.cfg.entity_type == 'baby-names':
        self.all_entities = utils.get_unique_names(split)
        self.all_entities = [
            common_types.Entity(entity, None) for entity in self.all_entities
        ]
      elif self.cfg.entity_type == 'random_name':
        self.congruency_generator = GenerateCongruentEntities(
            common_types.CongruencyMode.RANDOM_NAME, self.split
        )
    else:
      self.all_entities = self.cfg.entities_input
    self.true_ordering = copy.copy(self.all_entities)
    self.network_type = cfg.network_type

  def generate_module(
      self,
      prng_key: jax.Array,
      ignore_list_entities: Optional[list[common_types.Entity]] = None,
  ) -> ComparisonModule:
    """Generates a ComparisonModule which represents a whole Comparison example.

    Args:
      prng_key: A jax prng key used to generate the module. Do not use sources
        of randomization from other libraries except with this key and JAX.
      ignore_list_entities: List of entities to skip when generating the module.

    Returns:
      a comparison module
    """
    module = ComparisonModule()
    if self.cfg.entity_type == 'congruent_objects':
      if ignore_list_entities:
        raise NotImplementedError(
            'Congruency generator does not implement ignore list.'
        )
      prng_key, subkey = jax.random.split(prng_key)
      gen_entities_sorted = self.congruency_generator.generate_set_of_entities(
          subkey, self.cfg.num_entities_max, self.cfg.relation_type
      )
    else:
      # Determine the ordering
      if self.cfg.randomize_relations:
        # Randomize list of entities. First element MOST (biggest/tallest etc)
        prng_key, subkey = jax.random.split(prng_key)
        self.all_entities = utils.jax_permutation(subkey, self.all_entities)

      # Choose active entities
      if ignore_list_entities:
        entities_to_choose = list(
            set(self.all_entities) - set(ignore_list_entities)
        )
      else:
        entities_to_choose = self.all_entities
      assert self.cfg.num_entities_max <= len(entities_to_choose)

      prng_key, subkey = jax.random.split(prng_key)
      gen_entities_unsorted = utils.jax_sample(
          subkey, entities_to_choose, self.cfg.num_entities_max, replace=False
      )

      # Sort them by the ordering
      gen_entities_sorted = []
      for entity in self.true_ordering:
        if entity in gen_entities_unsorted:
          gen_entities_sorted.append(entity)

    module.entities = gen_entities_sorted
    if self.cfg.network_type == 'linear':
      # Generate blocks
      for i in range(1, len(gen_entities_sorted)):
        entity_1 = gen_entities_sorted[i - 1]
        entity_2 = gen_entities_sorted[i]
        prng_key, subkey = jax.random.split(prng_key)
        comparison_block = self.comparison_generator.generate(
            entity_1, entity_2, subkey
        )
        module.blocks.append(comparison_block)
        # Query block may be unused if the module is a distractor
        prng_key, subkey = jax.random.split(prng_key)
        query_block = generate_comparison_query_block(
            self.cfg, gen_entities_sorted, subkey, None
        )
        module.answer_block = query_block
    elif self.cfg.network_type == 'random_tree':
      n = self.cfg.num_entities_max
      # Sample a random number from jax and use it to seed the random tree.
      prng_key, subkey = jax.random.split(prng_key)
      nx_seed = jax.random.randint(subkey, (), 0, 10000000)
      g = nx.random_labeled_tree(n, seed=int(nx_seed))
      graph_triu = lambda g: nx.DiGraph(np.triu(nx.to_numpy_array(g)))
      digraph = graph_triu(g)
      for i in digraph.edges:
        entity_idx1, entity_idx2 = i[0], i[1]
        entity_1 = gen_entities_sorted[entity_idx1]
        entity_2 = gen_entities_sorted[entity_idx2]
        prng_key, subkey = jax.random.split(prng_key)
        comparison_block = self.comparison_generator.generate(
            entity_1, entity_2, subkey
        )
        module.blocks.append(comparison_block)
      query_block = generate_comparison_query_block(
          self.cfg, gen_entities_sorted, subkey, digraph
      )
      module.answer_block = query_block
    else:
      raise NotImplementedError(
          f'Need to implement this network type: {self.cfg.network_type}'
      )

    # Generate an ordering for the blocks
    if self.cfg.ordering == common_types.Ordering.INORDER.value:
      pass
    elif self.cfg.ordering == common_types.Ordering.REVERSE.value:
      module.blocks.reverse()
    elif self.cfg.ordering == common_types.Ordering.RANDOM.value:
      prng_key, subkey = jax.random.split(prng_key)
      module.blocks = utils.jax_permutation(subkey, module.blocks)
    return module

  def generate_validity_module(
      self,
      prng_key: jax.Array,
      ignore_list_entities: Optional[list[common_types.Entity]] = None,
  ) -> ComparisonCoherentEvidenceModule:
    """Generates a ComparisonModule which represents a whole Comparison example.

    Args:
      prng_key: A jax prng key used to generate the module. Do not use sources
        of randomization from other libraries except with this key and JAX.
      ignore_list_entities: List of entities to skip when generating the module.

    Returns:
      a comparison module
    """
    module = ComparisonCoherentEvidenceModule()
    # Determine the ordering
    if self.cfg.randomize_relations:
      # Randomize list of entities. First element MOST (biggest/tallest etc)
      prng_key, subkey = jax.random.split(prng_key)
      self.all_entities = utils.jax_permutation(subkey, self.all_entities)

    # Choose active entities
    if ignore_list_entities:
      entities_to_choose = list(
          set(self.all_entities) - set(ignore_list_entities)
      )
    else:
      entities_to_choose = self.all_entities
    assert self.cfg.num_entities_max <= len(entities_to_choose)

    prng_key, subkey = jax.random.split(prng_key)
    gen_entities_unsorted = utils.jax_sample(
        subkey, entities_to_choose, self.cfg.num_entities_max, replace=False
    )

    # Sort them by the ordering
    gen_entities_sorted = []
    for entity in self.true_ordering:
      if entity in gen_entities_unsorted:
        gen_entities_sorted.append(entity)

    module.entities = gen_entities_sorted

    n = self.cfg.num_entities_max
    # Sample a random number from jax and use it to seed the random tree.
    prng_key, subkey = jax.random.split(prng_key)
    nx_seed = jax.random.randint(subkey, (), 0, 10000000)
    g = nx.random_labeled_tree(n, seed=int(nx_seed))
    graph_triu = lambda g: nx.DiGraph(np.triu(nx.to_numpy_array(g)))
    digraph = graph_triu(g)
    for i in digraph.edges:
      entity_idx1, entity_idx2 = i[0], i[1]
      entity_1 = gen_entities_sorted[entity_idx1]
      entity_2 = gen_entities_sorted[entity_idx2]
      prng_key, subkey = jax.random.split(prng_key)
      comparison_block = self.comparison_generator.generate(
          entity_1, entity_2, subkey
      )
      module.blocks.append(comparison_block)
    prng_key, subkey = jax.random.split(prng_key)
    make_invalid = int(jax.random.randint(subkey, shape=(), minval=0, maxval=2))
    # invalidate the graph by adding at least 1 edge that induces
    # a cycle. We do this by randomly sampling nodes from the graph until
    # we find one where a path exists, then we add a directed edge against the
    # path.
    if not make_invalid:
      # Keep valid, add any edge that doesn't induce a cycle.
      prng_key, subkey = jax.random.split(prng_key)
      index_q_ent1, index_q_ent2 = jax.random.choice(
          subkey,
          jnp.arange(len(gen_entities_sorted)),
          replace=False,
          shape=(2,),
      )
      index_q_ent1 = int(index_q_ent1)
      index_q_ent2 = int(index_q_ent2)
      if not nx.has_path(digraph, index_q_ent2, index_q_ent1):
        digraph.add_edge(index_q_ent1, index_q_ent2, directed=True)
        entity1 = gen_entities_sorted[index_q_ent1]
        entity2 = gen_entities_sorted[index_q_ent2]
      else:
        digraph.add_edge(index_q_ent2, index_q_ent1, directed=True)
        entity1 = gen_entities_sorted[index_q_ent2]
        entity2 = gen_entities_sorted[index_q_ent1]
    else:
      num_retry_attempts = 10
      while True:
        prng_key, subkey = jax.random.split(prng_key)
        a, b = jax.random.choice(
            subkey,
            jnp.arange(len(gen_entities_sorted)),
            replace=False,
            shape=(2,),
        )
        a = int(a)
        b = int(b)
        # Must add only a valid edge.
        if nx.has_path(digraph, a, b):
          if num_retry_attempts < 0:
            break
          if nx.shortest_path_length(digraph, a, b) > 1:
            break
        if nx.has_path(digraph, b, a):
          if num_retry_attempts < 0:
            a, b = b, a
            break
          if nx.shortest_path_length(digraph, b, a) > 1:
            a, b = b, a
            break
        num_retry_attempts -= 1
      # Guarantee for a path to always exist from q1 to q2. Add path from q2
      # to q1 to generate a cycle.
      digraph.add_edge(b, a, directed=True)
      entity1 = gen_entities_sorted[b]
      entity2 = gen_entities_sorted[a]
    prng_key, subkey = jax.random.split(prng_key)
    comparison_block = self.comparison_generator.generate(
        entity1, entity2, subkey
    )
    module.blocks.append(comparison_block)
    query_block = recoglab_base.ReCogLabQuestionAnswerBlock()
    query_block.prompt.append(
        'Are the above statements consistent or inconsistent with each other?'
    )
    query_block.answers = ['Inconsistent' if make_invalid else 'Consistent']
    module.answer_block = query_block
    # Generate an ordering for the blocks
    if self.cfg.ordering == 'inorder':
      pass
    elif self.cfg.ordering == 'reverse':
      module.blocks.reverse()
    elif self.cfg.ordering == 'random':
      prng_key, subkey = jax.random.split(prng_key)
      module.blocks = utils.jax_permutation(subkey, module.blocks)

    return module
