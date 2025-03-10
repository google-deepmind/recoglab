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

"""Code for generating EEAAO dataset."""
from collections.abc import Iterable, Sequence
import functools
import os
from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
import ml_collections
import networkx as nx
import numpy as np

from recoglab import common_types
from recoglab import utils
from recoglab.modules import recoglab_base


SENTENCE_TEMPLATE = utils.get_resource_path('data/text_templates')


class SocialTextGenerator:
  """Generate setences that describe a particular relationship."""

  def __init__(self, filepath: str):
    """Loads a list of generated template sentences from a file."""
    self.filepath = filepath
    self.sentences = []

  def load_sentences(self) -> None:
    """Load sentence templates from file.

    This is done on demand instead of at initialization.
    """
    with open(self.filepath, 'r') as f:
      file_content = f.read()
    sentences = file_content.split('\n')
    self.sentences = [s.strip() for s in sentences if s.strip()]

  def join(self, entities_name: List[str], prng_key: jax.Array) -> str:
    """Entities is a list. Uses the same API as str.join.

    Args:
      entities_name: list of entities to insert into a random template sentence.
      prng_key: PRNGKey for sampling a sentence.

    Raises:
      ValueError: if entities_name is not length 2.

    Returns:
      A string with the entities inserted into a template.
    """
    if len(entities_name) != 2:
      raise ValueError('Only supports two entities')

    if not self.sentences:
      # Lazy load sentences.
      self.load_sentences()
    _, subkey = jax.random.split(prng_key)
    sentence_idx = jax.random.choice(subkey, jnp.arange(len(self.sentences)))
    sentence_template = self.sentences[sentence_idx]
    sentence_template = sentence_template.replace('[Person1]', entities_name[0])
    sentence_template = sentence_template.replace('[person1]', entities_name[0])
    sentence_template = sentence_template.replace('[Person2]', entities_name[1])
    sentence_template = sentence_template.replace('[person2]', entities_name[1])
    return sentence_template


@functools.cache
def get_basic_social_bridges() -> Dict[str, str | SocialTextGenerator]:
  """Get default social bridges."""
  return {
      'friend': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                 'friends_basic.txt')),
      'relative': ' is related to ',
      'knows': ' knows ',
      'class': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                'class_basic.txt')),
      'team': ' is on the same team as ',
      'parent': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                 'parents_basic.txt')),
      'sibling': ' is the sibling of ',
      'child': ' is the child of ',
      'spouse': ' is married to ',
      # negative/antagonistic relationships
      'enemies': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                  'enemy_basic.txt')),
  }


@functools.cache
def get_advanced_social_bridges() -> Dict[str, str | SocialTextGenerator]:
  """Get advanced social bridges."""
  return {
      'friend': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                 'friends_adv.txt')),
      'relative': ' is related to ',
      'knows': ' knows ',
      'class': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                'class_adv.txt')),
      'team': ' is on the same team as ',
      'parent': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                 'parents_adv.txt')),
      'sibling': ' is the sibling of ',
      'child': ' is the child of ',
      'spouse': ' is married to ',
      # negative/antagonistic relationships
      'enemies': SocialTextGenerator(os.path.join(SENTENCE_TEMPLATE,
                                                  'enemy_adv.txt')),
  }

SYM_SOCIAL_BRIDGES = [
    'friend', 'relative', 'knows', 'class', 'team', 'sibling', 'spouse']


def common_fastest_message(
    graph: nx.Graph | nx.DiGraph, query_idx: Sequence[int]
) -> Iterable[Sequence[int]]:
  """Generates shared block of all fastest_message questions.

  Args:
    graph: A nx graph object representing the social network.
    query_idx: Gather the shortest path between query_idx nodes.

  Returns:
    Returns an iterable of shortest paths between two randomly sampled nodes.
  """
  # Get shortest path.
  return nx.all_shortest_paths(graph, query_idx[0], query_idx[1])


def fastest_message_block(
    cfg: ml_collections.ConfigDict,
    prng_key: jax.Array,
    entities: List[common_types.Entity],
    graph: Union[nx.Graph, nx.DiGraph],
) -> recoglab_base.ReCogLabQuestionAnswerBlock:
  """Generate a query and answer for the Fastest Message question.

  Args:
    cfg: config with the following fields: query_preamble - preamble string
      which will be prepended to the generated query text.
    prng_key: PRNGKey for generating the query.
    entities: list of all entities in the module (query entities will be sampled
      from this list).
    graph: nx.Graph object representing the social network.

  Returns:
    a recoglab_base.ReCogLabQuestionAnswerBlock of a query/answer.
  """

  n = graph.number_of_nodes()
  _, subkey = jax.random.split(prng_key)
  query_idxs = jax.random.choice(
      subkey, jnp.arange(n), shape=(2,), replace=False
  )
  elem = range(n)
  query_idxs = [elem[i] for i in query_idxs]
  shortest_paths = list(common_fastest_message(graph, query_idxs))

  # First step on shortest path.
  all_valid_answers = [entities[sp[1]].text for sp in shortest_paths]
  entity_1 = entities[query_idxs[0]]
  entity_2 = entities[query_idxs[1]]

  query = cfg.query_preamble + (
      f'If {entity_1.text} wants to get a message to {entity_2.text} as'
      f' quickly as possible, who should {entity_1.text} give it to?'
  )
  block = recoglab_base.ReCogLabQuestionAnswerBlock()
  # Contains all valid answers.
  block.answers.extend(all_valid_answers)

  block.prompt.append(query)
  block.metadata.update({'symbolic_distance': str(len(shortest_paths[0]) - 1),
                         'entities': repr([e.text for e in entities])})
  return block


def fastest_message_block_num_hops(
    cfg: ml_collections.ConfigDict,
    prng_key: jax.Array,
    entities: Sequence[common_types.Entity],
    graph: Union[nx.Graph, nx.DiGraph],
) -> recoglab_base.ReCogLabQuestionAnswerBlock:
  """Creates a query answer for Fastest Message - Num Hops.

  Args:
    cfg: config with the following fields: query_preamble - preamble string
      which will be prepended to the generated query text.
    prng_key: PRNGKey for generating the query.
    entities: list of all entities in the module (query entities will be sampled
      from this list).
    graph: nx.Graph object representing the social network.

  Returns:
    a eeaao_base.EEAAOQuestionAnswerBlock of a query/answer.
  """
  n = graph.number_of_nodes()
  _, subkey = jax.random.split(prng_key)
  query_idxs = utils.jax_sample(subkey, tuple(range(n)), 2, replace=False)
  shortest_paths = list(common_fastest_message(graph, query_idxs))
  entity_1 = entities[query_idxs[0]]
  entity_2 = entities[query_idxs[1]]
  query = cfg.query_preamble + (
      'In numerical form, how many times does a message from'
      f' {entity_1.text} need to be exchanged to reach {entity_2.text} as'
      ' quickly as possible?'
  )
  block = recoglab_base.ReCogLabQuestionAnswerBlock()
  symbolic_distance = len(shortest_paths[0]) - 1
  block.answers.append(str(symbolic_distance))

  block.prompt.append(query)
  block.metadata.update({'symbolic_distance': str(len(shortest_paths[0]) - 1),
                         'entities': repr([e.text for e in entities])})
  return block


def fastest_message_block_exact_path(
    cfg: ml_collections.ConfigDict,
    prng_key: jax.Array,
    entities: Sequence[common_types.Entity],
    graph: Union[nx.Graph, nx.DiGraph],
) -> recoglab_base.ReCogLabQuestionAnswerBlock:
  """Creates a query answer for Fastest Message - Exact Paths.

  This requires formatting the answer as a Python list.

  Args:
    cfg: config with the following fields: query_preamble - preamble string
      which will be prepended to the generated query text.
    prng_key: PRNGKey for generating the query.
    entities: list of all entities in the module (query entities will be sampled
      from this list).
    graph: nx.Graph object representing the social network.

  Returns:
    a eeaao_base.EEAAOQuestionAnswerBlock of a query/answer.
  """
  n = graph.number_of_nodes()
  _, subkey = jax.random.split(prng_key)
  query_idxs = utils.jax_sample(subkey, list(range(n)), 2, replace=False)
  entity_1 = entities[query_idxs[0]]
  entity_2 = entities[query_idxs[1]]
  shortest_paths = list(common_fastest_message(graph, query_idxs))
  block = recoglab_base.ReCogLabQuestionAnswerBlock()
  for p in shortest_paths:
    python_list = str([entities[i].text for i in p])
    block.answers.append(python_list)
  query = cfg.query_preamble + (
      'What is the fastest path of people that a message from'
      f' {entity_1.text} to {entity_2.text}, would pass through? '
      'Please format your answer as a Python list including '
      f' {entity_1.text}, {entity_2.text}.'
  )

  block.prompt.append(query)
  block.metadata.update({'symbolic_distance': str(len(shortest_paths[0]) - 1),
                         'entities': repr([e.text for e in entities])})
  return block


def generate_query_young_old_block(
    prng_key: jax.Array,
    entities: Sequence[common_types.Entity],
    graph: Union[nx.Graph, nx.DiGraph],
    query_oldest: bool = True,
) -> recoglab_base.ReCogLabQuestionAnswerBlock:
  """Generate a query and answer for the Youngest/Oldest generation question.

  Args:
    prng_key: source of randomization.
    entities: list of all entities in the module (query entities will be sampled
      from this list).
    graph: nx.Graph object representing the social network.
    query_oldest: if true, asks a question about the oldest generation.
      Otherwise, asks about the youngest generation.

  Returns:
    a recoglab_base.ReCogLabQuestionAnswerBlock of a query/answer.

  Throws:
    DataGenerationError: If no common descendant is found.
  """
  block = recoglab_base.ReCogLabQuestionAnswerBlock()
  # Generate the graph and get highest ancestors.
  if not query_oldest:
    # The youngest generation reverses the DAG and queries for oldest.
    assert isinstance(graph, nx.DiGraph), 'Must be a DiGraph'
    working_graph = graph.reverse()
  else:
    working_graph = graph
  highest_ancestors = get_highest_ancestor_on_dag(working_graph)

  answer = [entities[o].text for o in highest_ancestors]
  _, subkey = jax.random.split(prng_key)
  random_member = utils.jax_sample(
      subkey, range(len(entities)), 1, replace=False
  )
  attribute = 'oldest' if query_oldest else 'youngest'
  query = (
      'The above statements are about family and we want to infer something'
      ' about this family.\n'
  )
  query += (
      f"Who is part of the {attribute} generation in %s's family tree?"
      % entities[random_member[0]].text
  )
  block.prompt.append(query)
  block.answers.extend(answer)
  del random_member
  return block


class SocialBlockGenerator:
  """Block describing a relationship between two people."""

  def __init__(self, cfg: Any):
    """Initialize.

    Args:
      cfg: config with the following fields: relation_type - describes the
        relationship b/t entity_1 and entity_2. Options include... 'random' -
        uniform random relationships sampled from SOCIAL_BRIDGES.
        'random_symmetric' - uniform random relationships that are symmetric,
        sampled from SYM_SOCIAL_BRIDGES (i.e. "A is parent of B" is not
        symmetric, because it does not imply that "B is parent of A". But "A is
        related to B" is symmetric.) 'karate' - "A does Karate with B" other -
        any string in SOCIAL_BRIDGES.keys() preamble - preamble string which
        will be prepended to the generated text. social_bridge - 'basic' or
        'advanced'. entity_type - 'baby-names', 'satc', or 'images'
    """
    self.cfg = cfg
    self.bridges = get_basic_social_bridges()
    self.bridges.update(
        {f'{k}_advanced': v for k, v in get_advanced_social_bridges().items()}
    )

  def generate(
      self,
      prng_key: jax.Array,
      entity_1: common_types.Entity,
      entity_2: common_types.Entity,
  ) -> recoglab_base.ReCogLabBlock:
    """Generate EEAABlock containing a relationship between entity_1 and 2.

    Args:
      prng_key: PRNGKey for generating the relationship.
      entity_1: NamedTuple with fields 'text' and 'image', name of first entity
        to appear in text ("A" in "A is related to B", and optionally image).
      entity_2: Namedtuple with fields 'text' and 'image', name of second entity
        to appear in text ("B" in "B is related to A", and optionally image.).

    Returns:
      A social block of the relationship between entity_1 and entity_2.
    """

    block = recoglab_base.ReCogLabBlock()
    bridge_advanced = self.bridges[self.cfg.relation_type]
    if isinstance(bridge_advanced, SocialTextGenerator):
      _, subkey = jax.random.split(prng_key)
      block.prompt = [
          self.cfg.preamble,
          bridge_advanced.join([entity_1.text, entity_2.text], subkey),
      ]
    else:
      block.prompt = [
          self.cfg.preamble,
          bridge_advanced.join([entity_1.text, entity_2.text]),
      ]
    return block


def higher_ancestor_on_dag(
    dg: nx.DiGraph,
    ancestor1: common_types.Entity,
    ancestor2: common_types.Entity,
) -> List[common_types.Entity]:
  """Returns the higher up of two ancestors to the first common descendant.

  If the two ancestors are equally distant from the common descendant,
  returns a list of both ancestors.

  Args:
    dg: nx.DiGraph representing the graph
    ancestor1: The first ancestor node
    ancestor2: The second ancestor node

  Returns:
    If the ancestors have no common descendant, returns an empty list.
    If the ancestors are equally distant from the common descendant,
    returns a list of both ancestors.
    Otherwise, returns the oldest ancestor.
  """
  desc1 = set(nx.descendants(dg, ancestor1))
  desc2 = set(nx.descendants(dg, ancestor2))
  x = list(desc1.intersection(desc2))
  if not x:
    # No common descendant.
    return []
  shortest_path_ancestor1 = min([
      len(path) for path in nx.all_simple_paths(
          dg, source=ancestor1, target=x[0])])
  shortest_path_ancestor2 = min([
      len(path) for path in nx.all_simple_paths(
          dg, source=ancestor2, target=x[0])])
  if shortest_path_ancestor1 < shortest_path_ancestor2:
    return [ancestor2]
  elif shortest_path_ancestor1 > shortest_path_ancestor2:
    return [ancestor1]
  return [ancestor1, ancestor2]


def get_highest_ancestor_on_dag(dag: nx.DiGraph) -> List[int]:
  """Returns a list of the highest ancestor in the dag.

  Args:
    dag: nx.DiGraph representing the graph

  Returns:
    A list of the highest ancestor in the dag.

  Raises:
    DataGenerationError: If no common descendant is found.
  """
  no_parent = []
  for v in nx.topological_sort(dag):
    if not nx.ancestors(dag, v):
      no_parent.append(v)

  if len(no_parent) == 1:
    # Only 1 source node so it must be the oldest.
    return no_parent
  larger_graph_edge = []
  for i, p in enumerate(no_parent):
    for _, p2 in enumerate(no_parent[i+1:]):
      answer = higher_ancestor_on_dag(dag, p, p2)
      if not answer:
        raise common_types.DataGenerationError('No common descendant')
      if answer[0] == p:
        larger_graph_edge.append((p, p2))
      elif answer[0] == p2:
        larger_graph_edge.append((p2, p))
  ancestors = []
  larger_graph = nx.DiGraph(larger_graph_edge)
  for v in larger_graph.nodes:
    if not nx.ancestors(larger_graph, v):
      ancestors.append(v)
  return ancestors


class SocialNetworkModule(recoglab_base.ReCogLabModule):
  """Module for social networks."""

  def __init__(self):
    super().__init__()
    self.graph = nx.Graph()
    self.digraph = nx.DiGraph()
    self.edges = []

  def get_adjacency_list(self) -> List[Dict[str, List[str]]]:
    """Returns the adjacency list for a given module."""
    n = self.graph.number_of_nodes()
    node2label = {i: self.entities[i].text for i in range(n)}
    adjlist = [
        {
            node2label[node]: [
                node2label[neighbor] for neighbor in self.graph.neighbors(node)
            ]
        }
        for node in self.graph.nodes
    ]

    return adjlist


class SocialNetworkModuleGenerator:
  """Generates a module of social relationships from a graph."""

  def __init__(
      self,
      cfg: ml_collections.ConfigDict,
      jax_init_key: jax.Array,
      split: common_types.DatasetSplit,
  ):
    """Initialize.

    Args:
      cfg: config with the following fields: entities_mode - 'preset' or
        'input'. If 'preset', entities will be sampled from a preset list of
        entities based on entity_type. If 'input', entities will be passed in as
        a list in cfg.entities_input. entities_input - list of entities to use.
        Only used if entities_mode is 'input'. entity_type - type of entities to
        use. Only used if entities_mode is 'preset'. Options include 'satc' (Sex
        and the City) and 'baby-names' (list of N baby names from
        1800s-present). num_entities_max - maximum number of entities to
        include. network_type - type of network to use. Options include
        'linear', 'random_tree', 'erdos_renyi', 'karate_club'. erdos_renyi -
        config with kwargs for nx.erdos_renyi_graph. randomize_relations - if
        True, shuffle order of the relations in module. randomize_direction - if
        True, randomly reverse the direction of the edges in the graph. any
        additional fields required by BlockGenerators
      jax_init_key: A jax key used to initialize the module. Do not use this
        PRNGKey in generate_value().
      split: The split to generate the module for.
    """
    self.cfg = cfg
    self.split = split
    # Creates query and answer block.

    # The initialization doesn't use randomization, but it's provided as an
    # argument in case future modulegenerators need randomization at
    # initialization. Follow all rules of jax.random.PRNGKey.
    del jax_init_key
    # Creates unit of information. Can optionally pass init key here.
    self.social_block_generator = SocialBlockGenerator(self.cfg)

    # Set the entities
    if self.cfg.entities_mode == common_types.EntityMode.PRESET.value:
      if self.cfg.entity_type == 'satc':
        self.all_entities = [
            'Samantha', 'Carrie', 'Charlotte', 'Miranda', 'Steve', 'Big']
        self.num_entities_max = 6
      elif self.cfg.entity_type == 'baby-names':
        self.all_entities = utils.get_unique_names(split=self.split)
        if self.cfg.num_entities_max is None:
          raise ValueError('cfg.num_entities_max must be set for baby-names')
        self.num_entities_max = self.cfg.num_entities_max
      else:
        raise ValueError(
            'cfg.entity_type must be "satc" or "baby-names", not'
            f' {self.cfg.entity_type}'
        )
    else:
      self.all_entities = self.cfg.entity_input
      if self.cfg.num_entities_max is None:
        raise ValueError('cfg.num_entities_max must be set for manual input')
      self.num_entities_max = self.cfg.num_entities_max
    self.all_entities = [
        common_types.Entity(text=e, image=None) for e in self.all_entities
    ]

  def generate_module(
      self,
      prng_key: jax.Array,
      ignore_list_entities: Optional[list[common_types.Entity]] = None,
  ) -> SocialNetworkModule:
    """Generate a module of social relationships of a graph.

    Args:
      prng_key: A jax prng key used to generate the module. Do not use sources
        of randomization from other libraries except with this key and JAX.
      ignore_list_entities: List of entities to skip when generating the module.

    Returns:
      A randomly constructed SocialNetworkModule

    Throws:
      DataGenerationError: If an invalid graph is generated.
    """
    module = SocialNetworkModule()
    if ignore_list_entities is None:
      ignore_list_entities = []
    entities_to_choose = [
        v for v in self.all_entities if v not in ignore_list_entities
    ]
    choose_n = self.num_entities_max
    prng_key, subkey = jax.random.split(prng_key)
    sampled_indices = jax.random.choice(
        subkey,
        jnp.arange(len(entities_to_choose)),
        replace=False,
        shape=(choose_n,),
    )
    module.add_entity([entities_to_choose[i] for i in sampled_indices])
    # Construct social networks
    n = len(module.entities)
    graph_triu = lambda g: nx.DiGraph(np.triu(nx.to_numpy_array(g)))
    if self.cfg.network_type == 'linear':
      g = nx.grid_graph((n,))
    elif self.cfg.network_type == 'random_tree':
      # Sample a random number from jax and use it to seed the random tree.
      prng_key, subkey = jax.random.split(prng_key)
      nx_seed = jax.random.randint(subkey, (), 0, 10000000)
      g = nx.random_labeled_tree(n, seed=int(nx_seed))
    elif self.cfg.network_type == 'erdos_renyi':
      g = nx.erdos_renyi_graph(n, self.cfg.erdos_renyi.p)
    elif self.cfg.network_type == 'karate_club':
      g = nx.karate_club_graph()
      self.entities = module.entities[: g.number_of_nodes()]
    else:
      raise NotImplementedError(
          f'Need to implement this network type: {self.cfg.network_type}')

    module.graph = g
    # keep only (i, j) not (j, i) edges
    module.edges = list(graph_triu(g).edges)

    # Randomize edge direction and order.
    if self.cfg.randomize_direction:
      p = 0.5
    else:
      p = 0.0
    prng_key, subkey = jax.random.split(prng_key)
    flip_edges = jax.random.uniform(subkey, shape=(len(g.edges),))
    module.edges = [
        edge if flip_edges[idx] < p else (edge[1], edge[0])
        for idx, edge in enumerate(g.edges)
    ]

    for e in module.edges:
      module.digraph.add_edge(*e)

    # Determine the ordering
    if self.cfg.randomize_relations:
      prng_key, subkey = jax.random.split(prng_key)
      permute_indices = jax.random.permutation(
          subkey, jnp.arange(len(module.edges))
      )
      module.edges = [module.edges[i] for i in permute_indices]

    # Generate blocks
    blocks = []
    for i, j in module.edges:
      prng_key, subkey = jax.random.split(prng_key)
      block = self.social_block_generator.generate(
          subkey, module.entities[i], module.entities[j]
      )
      blocks.append(block)
    module.add_block(blocks)

    # Generate Query and Answer block for a specific task.
    if self.cfg.query_task == 'FastestMessage':
      _, subkey = jax.random.split(prng_key)
      query_block = fastest_message_block(
          self.cfg,
          subkey,
          module.entities,
          module.graph,
      )
    elif self.cfg.query_task == 'FastestMessage_NumHops':
      _, subkey = jax.random.split(prng_key)
      query_block = fastest_message_block_num_hops(
          self.cfg,
          subkey,
          module.entities,
          module.graph,
      )
    elif self.cfg.query_task == 'FastestMessage_ExactPath':
      _, subkey = jax.random.split(prng_key)
      query_block = fastest_message_block_exact_path(
          self.cfg,
          subkey,
          module.entities,
          module.graph,
      )
    elif self.cfg.query_task == 'OldestGeneration':
      _, subkey = jax.random.split(prng_key)
      query_block = generate_query_young_old_block(
          subkey, module.entities, module.digraph, query_oldest=True
      )
    elif self.cfg.query_task == 'YoungestGeneration':
      _, subkey = jax.random.split(prng_key)
      query_block = generate_query_young_old_block(
          subkey, module.entities, module.digraph, query_oldest=False
      )
    else:
      raise NotImplementedError(
          f'Need to implement this query task: {self.cfg.query_task}'
      )
    module.answer_block = query_block
    return module
