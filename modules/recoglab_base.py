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

"""Code for generating ReCogLab dataset."""

from collections.abc import Iterable
from typing import Any, Dict, List, Set, Union

from recoglab import common_types


# Some basic block types
class ReCogLabBlock:
  """Data class for a block of text for the ReCogLab dataset.

  This should be lightweight with no generation logic. If you want to add
  some lightweight manipulation, extend from this module.

  Attributes:
    prompt: The text/image of the block
    prompt_separator: (Optional) String separator between prompt blocks.
    metadata: dict of string to string encoding information during the
      generation of this block.
  """

  def __init__(self):
    self.prompt = []
    self.prompt_separator = ''

    # This stores information about the generation process in case we want
    # to slice performance by specific attributes of the generation.
    self.metadata = {}

  def get_prompt(self, prompt_type: str) -> List[str]:
    # Gets the textual information of this block.
    raise NotImplementedError('Need to implement this block')

  def __repr__(self) -> str:
    """A string representation of a block."""
    return self.prompt_separator.join(self.prompt)


class ReCogLabQuestionAnswerBlock(ReCogLabBlock):
  """Data class that manages a question and answer."""

  def __init__(self):
    super().__init__()
    self.answers = []

  def get_accuracy(self, pred: str | List[str]) -> float:
    """Scores the prediction against the answers.

    Args:
      pred: Entity we are scoring as an answer.

    Returns:
      A float score of the prediction.
    """
    if not self.answers:
      raise NotImplementedError('self.answers needs to have answers')
    if pred in self.answers:
      score = 1
    else:
      score = 0
    return float(score)

  def get_question(self) -> str:
    """The question is in self.prompt so we use ReCogLabBlock repr."""
    return super().__repr__()

  def __repr__(self) -> str:
    """A string representation of a block."""
    prompt_repr = self.prompt_separator.join(self.prompt)

    if self.answers:
      answer_repr = ','.join(self.answers)
      return f'Prompt:{prompt_repr}\nAnswers: {answer_repr}'
    else:
      return prompt_repr


class ReCogLabModule:
  """A module is a sequence of highly related blocks.

  Please use this class for holding data, these are more or less
  equivalent to protos and should only contain information that can be
  replaced with a proto. Lightweight manipulation logic is okay. Generation
  logic should never go in any of these functions.

  Attributes:
    cfg: config for the module
    blocks: List of blocks, each of which describes a relationship, which
      collectively comprise a module of relationships.
    entities: List of entities that are used in the module.
    answer_block: a block that contains a query_block and answer based on the
      module.
  """

  def __init__(self):
    self.blocks = []
    self.entities = []
    self.answer_block: None | ReCogLabQuestionAnswerBlock = None

  def get_blocks(self) -> List[ReCogLabBlock]:
    return self.blocks

  def add_block(self, block: Iterable[ReCogLabBlock] | ReCogLabBlock) -> None:
    if isinstance(block, Iterable):
      self.blocks.extend(block)
    else:
      self.blocks.append(block)

  def get_answer_block(self) -> ReCogLabQuestionAnswerBlock:
    assert isinstance(
        self.answer_block, ReCogLabQuestionAnswerBlock
    ), 'Need to implement this answer block'
    return self.answer_block

  def get_entities(self) -> Set[Any]:
    return set(self.entities)

  def add_entity(
      self, entities: Union[List[common_types.Entity], common_types.Entity]
  ):
    if isinstance(entities, list):
      self.entities.extend(entities)
    else:
      self.entities.append(entities)

  def get_metadata(self) -> Dict[str, str]:
    metadata = {}
    for idx, b in enumerate(self.blocks):
      metadata.update({f'{idx}/{k}': v for k, v in b.metadata.items()})
    if self.answer_block is not None:
      metadata.update(
          {f'answer/{k}': v for k, v in self.answer_block.metadata.items()}
      )
    return metadata
