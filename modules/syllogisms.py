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

"""Code for generating syllogisms in ReCogLab framework."""

from collections.abc import Callable
import enum
from typing import Iterable, Iterator, Optional

import jax
import ml_collections
import nltk
import textblob

from recoglab import common_types
from recoglab import utils
from recoglab.modules import plural_nouns
from recoglab.modules import recoglab_base


wn = nltk.corpus.wordnet


class PropositionType(enum.Enum):
  # See https://en.wikipedia.org/wiki/Syllogism#Types for definitions
  A = enum.auto()
  E = enum.auto()
  I = enum.auto()
  O = enum.auto()


Type = PropositionType


PROPOSITION_TYPE_TO_PATTERN = {
    PropositionType.A: "All {S} are {P}",
    PropositionType.E: "No {S} are {P}",
    PropositionType.I: "Some {S} are {P}",
    PropositionType.O: "Some {S} are not {P}",
}


def _find_non_overlapping(noun: str, prng_key: jax.Array) -> str:
  """Finds a non-overlapping noun for a given noun.

  This function allows us to find "No A are B" WordNet relationships between two
  nouns. If the lowest common hypernym of two nouns is one of those nouns, then
  the desired relationship is invalid. We find a non-overlapping noun by
  randomly permuting the list of nouns and checking if the lowest common
  hypernym of the noun and the permuted noun is not one of the two nouns.

  Args:
    noun: the noun to find a non-overlapping noun for.
    prng_key: jax random key for randomization.

  Returns:
    A non-overlapping noun.
  """
  permuted_nouns = utils.jax_permutation(prng_key, plural_nouns.PLURAL_NOUNS)
  wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
  for other_noun in permuted_nouns:
    singular_other_noun = wordnet_lemmatizer.lemmatize(other_noun)
    singular_other_noun = _PLURAL_TO_SINGULAR[singular_other_noun]
    if noun == singular_other_noun:
      continue
    lowest_common_hypernym = wn.synset(f"{noun}.n.01").lowest_common_hypernyms(
        wn.synset(f"{singular_other_noun}.n.01")
    )
    if all(
        str(hypernym) not in [f"{noun}.n.01", f"{singular_other_noun}.n.01"]
        for hypernym in lowest_common_hypernym
    ):
      return other_noun
  raise ValueError(f"No non-overlapping noun found for {noun}")


def _extract_word(synset: nltk.corpus.reader.wordnet.Synset) -> str:
  """Returns the string word of a WordNet synset."""
  return [l.name() for l in synset.lemmas()][0]


class MissingDict(dict):

  def __missing__(self, key):
    return key


_PLURAL_TO_SINGULAR = MissingDict()


def extract_and_pluralize(synset: nltk.corpus.reader.wordnet.Synset) -> str:
  word = _extract_word(synset)
  plural = textblob.TextBlob(word).words.pluralize()[0]

  if not isinstance(plural, str):
    raise ValueError(f"Pluralization failed for {word}")

  _PLURAL_TO_SINGULAR[plural] = word
  return plural


def make_congruent(
    noun: str,
    proposition_type: PropositionType,
    is_subject: bool,
    prng_key: jax.Array,
) -> str:
  """Makes a congruent proposition for a given noun.

  Makes a congruent proposition given a noun, the desired proposition type, and
  whether the noun is meant to be the subject or predicate.

  Args:
    noun: the noun to make a congruent proposition for.
    proposition_type: the type of proposition to make.
    is_subject: whether the noun is the subject or predicate.
    prng_key: jax random key for randomization.

  Returns:
    A congruent proposition or an empty string if it is impossible.
  """
  wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
  singular_noun = wordnet_lemmatizer.lemmatize(noun)
  singular_noun = _PLURAL_TO_SINGULAR[singular_noun]
  noun_wordnet = wn.synset(f"{singular_noun}.n.01")
  if is_subject:
    if proposition_type == PropositionType.A:
      if hypernyms := noun_wordnet.hypernyms():
        return extract_and_pluralize(hypernyms[0])
    if proposition_type == PropositionType.E:
      return _find_non_overlapping(singular_noun, prng_key)
    if proposition_type == PropositionType.I:
      if hyponyms := noun_wordnet.hyponyms():
        return extract_and_pluralize(hyponyms[0])
    if proposition_type == PropositionType.O:
      if hyponyms := noun_wordnet.hyponyms():
        return extract_and_pluralize(hyponyms[0])
  else:
    if proposition_type == PropositionType.A:
      if hyponyms := noun_wordnet.hyponyms():
        return extract_and_pluralize(hyponyms[0])
    if proposition_type == PropositionType.E:
      return _find_non_overlapping(singular_noun, prng_key)
    if proposition_type == PropositionType.I:
      if hypernyms := noun_wordnet.hypernyms():
        return extract_and_pluralize(hypernyms[0])
    if proposition_type == PropositionType.O:
      if hypernyms := noun_wordnet.hypernyms():
        return extract_and_pluralize(hypernyms[0])
  return ""


def make_incongruent(
    noun: str,
    proposition_type: PropositionType,
    is_subject: bool,
    prng_key: jax.Array,
) -> str:
  """Makes an incongruent proposition for a given noun.

  Makes an incongruent proposition given a noun, the desired proposition type,
  and whether the noun is meant to be the subject or predicate.

  Args:
    noun: the noun to make an incongruent proposition for.
    proposition_type: the type of proposition to make.
    is_subject: whether the noun is the subject or predicate.
    prng_key: jax random key for randomization.

  Returns:
    An incongruent proposition or an empty string if it is impossible.
  """
  wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
  singular_noun = wordnet_lemmatizer.lemmatize(noun)
  singular_noun = _PLURAL_TO_SINGULAR[singular_noun]
  noun_wordnet = wn.synset(f"{singular_noun}.n.01")

  if is_subject:
    if proposition_type == PropositionType.A:
      if hyponyms := noun_wordnet.hyponyms():
        return extract_and_pluralize(hyponyms[0])
    if proposition_type == PropositionType.E:
      if hyponyms := noun_wordnet.hyponyms():
        return extract_and_pluralize(hyponyms[0])
    if proposition_type == PropositionType.I:
      return _find_non_overlapping(singular_noun, prng_key)
    if proposition_type == PropositionType.O:
      if hypernyms := noun_wordnet.hypernyms():
        return extract_and_pluralize(hypernyms[0])
  else:
    if proposition_type == PropositionType.A:
      if hypernyms := noun_wordnet.hypernyms():
        return extract_and_pluralize(hypernyms[0])
    if proposition_type == PropositionType.E:
      if hypernyms := noun_wordnet.hypernyms():
        return extract_and_pluralize(hypernyms[0])
    if proposition_type == PropositionType.I:
      return _find_non_overlapping(singular_noun, prng_key)
    if proposition_type == PropositionType.O:
      if hyponyms := noun_wordnet.hyponyms():
        return extract_and_pluralize(hyponyms[0])
  return ""


class PropositionBlock(recoglab_base.ReCogLabQuestionAnswerBlock):
  """Block for proposition of a syllogism.

  Attributes:
    subject: the subject of the proposition.
    predicate: the predicate of the proposition.
    proposition_type: the type of the proposition.
  """

  def __init__(self, subject: str, predicate: str, proposition_type: Type):
    super().__init__()
    self.subject = subject
    self.predicate = predicate
    self.proposition_type = proposition_type
    self.prompt = [
        PROPOSITION_TYPE_TO_PATTERN[proposition_type].format(
            S=subject, P=predicate
        )
    ]
    self.answers = self.prompt

  @property
  def subject_with_spaces(self) -> str:
    return " ".join(self.subject.split("_"))

  @property
  def predicate_with_spaces(self) -> str:
    return " ".join(self.predicate.split("_"))

  def set_as_conclusion(self) -> None:
    # When used as a conclusion, the prompt is all possible propositions with
    # the subject and predicate
    self.prompt = ["Which of the following is true?"]
    self.prompt.extend([
        PROPOSITION_TYPE_TO_PATTERN[p].format(
            S=self.subject_with_spaces, P=self.predicate_with_spaces
        )
        for p in PropositionType
    ])
    self.prompt_separator = "\n"

  @property
  def entities(self):
    return [self.subject, self.predicate]

  @property
  def is_affirmative(self):
    return self.proposition_type in [Type.A, Type.I]

  @property
  def is_negative(self):
    return self.proposition_type in [Type.E, Type.O]

  @property
  def is_universal(self):
    return self.proposition_type in [Type.A, Type.E]

  @property
  def is_particular(self):
    return self.proposition_type in [Type.I, Type.O]

  def __str__(self):
    return PROPOSITION_TYPE_TO_PATTERN[self.proposition_type].format(
        S=self.subject_with_spaces, P=self.predicate_with_spaces
    )


class SyllogismBlock(recoglab_base.ReCogLabBlock):
  """Block for defining syllogisms and checking their validity."""

  def __init__(
      self,
      premise_1: PropositionBlock,
      premise_2: PropositionBlock,
      conclusion: PropositionBlock,
  ):
    super().__init__()
    self.entities = list({
        *premise_1.entities,
        *premise_2.entities,
        *conclusion.entities,
    })
    if len(self.entities) != 3:
      raise ValueError("Syllogism must have three unique terms")
    self._premise_1 = premise_1
    self._premise_2 = premise_2
    self._conclusion = conclusion
    # Assign major and minor: if invalid, this won't matter
    if (
        premise_1.subject == conclusion.subject
        or premise_1.predicate == conclusion.subject
    ):
      self._minor = premise_1
      self._major = premise_2
    else:
      self._minor = premise_2
      self._major = premise_1

    self.prompt = [f"{self._premise_1}\n{self._premise_2}\nConclusion: "]
    self.answers = [str(self._conclusion)]

  @property
  def conclusion(self) -> PropositionBlock:
    return self._conclusion

  def figure(self):
    minor = self._conclusion.subject
    major = self._conclusion.predicate
    if self._minor.subject == minor:  # ??/S?/SP
      if self._major.subject == major:  # P?/S?/SP
        if self._major.predicate == self._minor.predicate:  # PM/SM/SP
          return 2
      elif self._major.predicate == major:  # ?P/S?/SP
        if self._major.subject == self._minor.predicate:  # MP/SM/SP
          return 1
    elif self._minor.predicate == minor:  # ??/?S/SP
      if self._major.subject == major:  # P?/?S/SP
        if self._major.predicate == self._minor.subject:  # PM/MS/SP
          return 4
      elif self._major.predicate == major:  # ?P/?S/SP
        if self._major.subject == self._minor.subject:  # MP/MS/SP
          return 3
    return 0

  def valid(self):
    # Three terms
    if self.figure() == 0:
      return False
    # Two negatives
    if self._minor.is_negative and self._major.is_negative:
      return False
    # Follow weaker
    if self._major.is_negative or self._minor.is_negative:
      if not self._conclusion.is_negative:
        return False
    if self._major.is_particular or self._minor.is_particular:
      if not self._conclusion.is_particular:
        return False
    # Middle distributed
    if self.figure() == 1:
      if not self._major.is_universal or self._minor.is_negative:
        return False
    elif self.figure() == 2:
      if not self._major.is_negative or self._minor.is_negative:
        return False
    elif self.figure() == 3:
      if not self._major.is_universal or self._minor.is_universal:
        return False
    elif self.figure() == 4:
      if not self._major.is_negative or self._minor.is_universal:
        return False
    # Terms distributed
    if self._conclusion.is_universal:
      if self.figure() in [1, 2]:
        if not self._minor.is_universal:
          return False
      elif self.figure() in [3, 4]:
        if not self._minor.is_negative:
          return False
    if self._conclusion.is_negative:
      if self.figure() in [1, 3]:
        if not self._major.is_negative:
          return False
      elif self.figure() in [2, 4]:
        if not self._major.is_universal:
          return False
    return True

  def possible_conclusions(self) -> Iterable[PropositionBlock]:
    for i in range(len(self.entities)):
      for j in range(i + 1, len(self.entities)):
        entity1 = self.entities[i]
        entity2 = self.entities[j]
        for proposition_type in PropositionType:
          yield PropositionBlock(entity1, entity2, proposition_type)
          yield PropositionBlock(entity2, entity1, proposition_type)

  def __str__(self):
    return f"{self.prompt}{self.conclusion}"


def naive_maybe_make_valid_conclusion(
    premise1: PropositionBlock, premise2: PropositionBlock
) -> SyllogismBlock | None:
  """Given two premises, returns a valid syllogism if one exists.

  Args:
    premise1: the first premise of the syllogism.
    premise2: the second premise of the syllogism.

  Returns:
    a valid proposition block if one exists.

  Raises:
    ValueError: if the syllogism does not have three unique terms.
  """
  entities = set([*premise1.entities, *premise2.entities])
  if len(entities) != 3:
    return None

  for subject in entities:
    for predicate in entities:
      for proposition_type in PropositionType:
        test_conclustion = PropositionBlock(
            subject, predicate, proposition_type
        )
        test_syllogism = SyllogismBlock(premise1, premise2, test_conclustion)
        if test_syllogism.valid():
          return test_syllogism


def _tree_search_chains_helper(
    entities: list[str],
    premises: list[PropositionBlock],
    current_conclusion: PropositionBlock,
    prng_key: jax.Array,
    get_entity_fn: (
        Callable[[str, PropositionType, bool, jax.Array], str] | None
    ) = None,
) -> Iterator[tuple[list[PropositionBlock], PropositionBlock]]:
  """Generates a syllogism chain from a starting premise and conclusion.

  Produces a generator of syllogism chains. Each chain is a tuple of
  premises and a conclusion. The premises are a list of PropositionBlocks
  and the conclusion is a PropositionBlock.

  Args:
    entities: the list of entities to use in the syllogism chain.
    premises: the list of premises in the syllogism chain.
    current_conclusion: the current conclusion in the syllogism chain.
    prng_key: jax random key for randomization.
    get_entity_fn: function to get an entity given a subject, predicate,
      proposition type, and a random key. If None, uses the next entity in the
      list of entities. Optional passed

  Yields:
    A tuple of premises and a conclusion.
  """
  if len(premises) + 1 == len(entities):
    yield premises[:], current_conclusion
    return
  prng_key, subkey = jax.random.split(prng_key)
  for t in utils.jax_sample(
      subkey, list(PropositionType), len(PropositionType)
  ):
    if get_entity_fn is not None:
      prng_key, *subkeys = jax.random.split(prng_key, 5)
      ordered_entity_pairs = []
      if next_entity := get_entity_fn(
          current_conclusion.subject, t, True, subkeys[0]
      ):
        ordered_entity_pairs.append((current_conclusion.subject, next_entity))
      if next_entity := get_entity_fn(
          current_conclusion.subject, t, False, subkeys[1]
      ):
        ordered_entity_pairs.append((next_entity, current_conclusion.subject))
      if next_entity := get_entity_fn(
          current_conclusion.predicate, t, True, subkeys[2]
      ):
        ordered_entity_pairs.append((current_conclusion.predicate, next_entity))
      if next_entity := get_entity_fn(
          current_conclusion.predicate, t, False, subkeys[3]
      ):
        ordered_entity_pairs.append((next_entity, current_conclusion.predicate))
    else:
      next_entity = entities[len(premises) + 1]
      ordered_entity_pairs = [
          (current_conclusion.subject, next_entity),
          (next_entity, current_conclusion.subject),
          (current_conclusion.predicate, next_entity),
          (next_entity, current_conclusion.predicate),
      ]
    for ordered_entity_pair in utils.jax_sample(
        subkey, ordered_entity_pairs, len(ordered_entity_pairs)
    ):
      candidate_premise = PropositionBlock(*ordered_entity_pair, t)
      candidate_syllogism = naive_maybe_make_valid_conclusion(
          current_conclusion, candidate_premise
      )
      if candidate_syllogism is not None:
        premises.append(candidate_premise)
        yield from _tree_search_chains_helper(
            entities,
            premises,
            candidate_syllogism.conclusion,
            prng_key,
            get_entity_fn,
        )
        premises.pop()


def tree_search_chains_factory(
    entities: list[str],
    get_entity_fn: (
        Callable[[str, PropositionType, bool, jax.Array], str] | None
    ) = None,
) -> Callable[
    [jax.Array], Iterator[tuple[list[PropositionBlock], PropositionBlock]]
]:
  """Factory for a generator of all valid syllogism chains from a list of entities.

  Args:
    entities: the list of entities to use in the syllogism chain.
    get_entity_fn: function to get an entity given a subject, predicate,
      proposition type, and a random key. If None, uses the next entity in the
      list of entities.

  Returns:
    A function that takes a PRNG key and returns an iterator of syllogism
    chains. Each chain is a tuple of premises and a conclusion. The premises
    are a list of PropositionBlocks and the conclusion is a PropositionBlock.
  """

  def _build_chain_generator(
      prng_key: jax.Array,
  ) -> Iterator[tuple[list[PropositionBlock], PropositionBlock]]:
    nonlocal entities
    prng_key, subkey = jax.random.split(prng_key)
    entities = utils.jax_permutation(subkey, entities)
    prng_key, subkey = jax.random.split(prng_key)
    for starting_type in utils.jax_sample(
        subkey, list(PropositionType), len(PropositionType)
    ):
      if get_entity_fn is not None:
        prng_key, subkey = jax.random.split(prng_key)
        next_entity = get_entity_fn(entities[0], starting_type, True, subkey)
        if not next_entity:
          continue
        starting_premise = PropositionBlock(
            entities[0], next_entity, starting_type
        )
      else:
        starting_premise = PropositionBlock(
            entities[0], entities[1], starting_type
        )
      yield from _tree_search_chains_helper(
          entities,
          [starting_premise],
          starting_premise,
          prng_key,
          get_entity_fn,
      )

  return _build_chain_generator


def _sort_premises(
    premises: list[PropositionBlock],
    conclusion_subject: str,
    conclusion_predicate: str,
) -> list[PropositionBlock]:
  """Sorts the premises so that entities are adjacently ordered.

  Constructs a Hamiltonian path to find a valid ordering of the premises.

  Args:
    premises: the list of premises to sort.
    conclusion_subject: the subject of the conclusion.
    conclusion_predicate: the predicate of the conclusion.

  Returns:
    A list of premises that are ordered such that entities are adjacently
    ordered.
  """
  indexed_premises = list(enumerate(premises))
  n = len(premises)

  adjacency = {i: [] for i in range(n)}

  for i, premise1 in indexed_premises:
    for j, premise2 in indexed_premises:
      if i == j:
        continue
      if set(premise1.entities) & set(premise2.entities):
        adjacency[i].append(j)

  starting_indices = [
      i
      for i, premise in indexed_premises
      if conclusion_subject in premise.entities
  ]
  ending_indices = [
      i
      for i, premise in indexed_premises
      if conclusion_predicate in premise.entities
  ]

  if not starting_indices or not ending_indices:
    raise ValueError("No starting or ending indices found")

  def dfs(current_idx: int, path, used):
    if len(path) == n:
      if conclusion_predicate in premises[current_idx].entities:
        return path
      else:
        return

    current_premise = premises[current_idx]
    for next_idx in adjacency[current_idx]:
      if next_idx in used:
        continue
      if not (set(current_premise.entities) & set(premises[next_idx].entities)):
        continue
      result = dfs(next_idx, path + [next_idx], used | {next_idx})
      if result is not None:
        return result

  for start_idx in starting_indices:
    result = dfs(start_idx, [start_idx], {start_idx})
    if result is not None:
      assert conclusion_predicate in premises[result[-1]].entities
      return [premises[i] for i in result]
  raise ValueError("No valid ordering found")


class SyllogismModule(recoglab_base.ReCogLabModule):
  """Module for a chain of syllogisms."""

  def __init__(self, entities: str):
    super().__init__()
    self.entities = entities


class SyllogismModuleGenerator:
  """Generates a module of syllogisms."""

  def __init__(
      self,
      cfg: ml_collections.ConfigDict,
      init_key: jax.Array,
      split: common_types.DatasetSplit,
  ):  # pylint: disable=g-doc-args
    """Initialize.

    Args:
      cfg: config with the following fields:
        entities_mode: 'preset' or 'input'. If 'preset', entities will be
          sampled from a preset list of entities based on entity_type. If
          'input', entities will be passed in as a list in cfg.entities_input.
        entities_input: list of entities to use. Only used if entities_mode is
          'input'.
        entity_type: type of entities to use. Only used if entities_mode is
          'preset'. Options include 'plural_nouns' (list of random nouns).
        num_entities_max: maximum number of entities to include.
      init_key: PRNG key for randomization.
      split: the split to generate the module for.
    """
    del init_key
    self.cfg = cfg
    self.split = split

    self.get_entity_fn = None
    if self.cfg.entities_mode == common_types.EntityMode.PRESET.value:
      if self.cfg.entity_type == "plural_nouns":
        self.all_entities = utils.get_unique_entities(
            self.cfg.entity_type, split=self.split
        )[: self.cfg.num_entities_max]
      else:
        raise ValueError(f"Unsupported entity type: {self.cfg.entity_type}")
    elif self.cfg.entities_mode == common_types.EntityMode.INPUT.value:
      self.all_entities = self.cfg.entities_input
    elif self.cfg.entities_mode == common_types.EntityMode.CONGRUENT.value:
      # Only first one is used, rest are generated based on congruence
      self.all_entities = utils.get_unique_entities(
          self.cfg.entity_type, split=self.split
      )[: self.cfg.num_entities_max]
      self.get_entity_fn = make_congruent
    elif self.cfg.entities_mode == common_types.EntityMode.INCONGRUENT.value:
      # Only first one is used, rest are generated based on incongruence
      self.all_entities = utils.get_unique_entities(
          self.cfg.entity_type, split=self.split
      )[: self.cfg.num_entities_max]
      self.get_entity_fn = make_incongruent
    else:
      raise ValueError(f"Unsupported entities mode: {self.cfg.entities_mode}")

    self._build_chain_generator = tree_search_chains_factory(
        self.all_entities, self.get_entity_fn
    )

  def generate_all_syllogisms(
      self, prng_key: jax.Array
  ) -> Iterator[SyllogismBlock]:
    """Returns a generator of all valid syllogims with `self.entities`."""
    yield from self._build_chain_generator(prng_key)

  def generate_module(
      self,
      prng_key: jax.Array,
      ignore_list_entities: Optional[list[common_types.Entity]] = None,
  ) -> SyllogismModule:
    """Generates a module of syllogisms."""
    if ignore_list_entities:
      entities_to_use = self.all_entities - ignore_list_entities
    else:
      entities_to_use = self.all_entities
    module = SyllogismModule(entities_to_use)
    chain_generator = self._build_chain_generator(prng_key)
    premises, conclusion = next(chain_generator)
    conclusion.set_as_conclusion()
    premises = _sort_premises(
        premises, conclusion.subject, conclusion.predicate
    )

    if premises[0].subject != conclusion.subject:
      premises = premises[::-1]

    if self.cfg.ordering == common_types.Ordering.INORDER.value:
      pass
    elif self.cfg.ordering == common_types.Ordering.REVERSE.value:
      premises = premises[::-1]
    elif self.cfg.ordering == common_types.Ordering.RANDOM.value:
      _, subkey = jax.random.split(prng_key)
      premises = utils.jax_permutation(subkey, premises)
    else:
      raise ValueError(f"Unsupported ordering: {self.cfg.ordering}")

    module.add_block(premises)
    module.answer_block = conclusion
    return module
