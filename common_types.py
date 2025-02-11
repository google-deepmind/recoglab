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

"""Standard type definitions shared by modules."""
import dataclasses
import enum
import re
from typing import Any

import immutabledict


# ==== Parsing Types and Helper Functions ====
UNKNOWN_ANSWER_DEFAULT_STRING = "UNKNOWN_ANSWER_DEFAULT_STRING"


class ParsingFunctionEnum(enum.StrEnum):
  STRIP_ONLY = "STRIP_ONLY"
  REGEX_NAIVE1 = "REGEX_NAIVE1"
  CLEAN_MODEL_RESPONSE = "CLEAN_MODEL_RESPONSE"
  BASE = "BASE"
  SYLLOGISM_CLEAN_MODEL_RESPONSE = "SYLLOGISM_CLEAN_MODEL_RESPONSE"
  PARSE_PYTHON_CODE_LIST = "PARSE_PYTHON_CODE_LIST"


def _regex_naive1(answer):
  """Uses regex search to parse for the first answer.

  Args:
    answer: The raw llm prediction.

  Returns:
    The parsed answer.
  """
  # Note that newline search must occur before the no newline search.
  # This is so Answer:\n<answer> is matched before checking Answer:<answer>.
  # The second one will trigger if the first one does, but the second one will
  # only return an empty string.
  answer_search1 = re.search(r"Answer:\n(.*)", answer)
  answer_search2 = re.search(r"\*\*Answer:\*\*\n(.*)", answer)
  answer_search3 = re.search(r"Answer:(.*)", answer)
  if answer_search1:
    answer = answer_search1.group(1)
  elif answer_search2:
    answer = answer_search2.group(1)
  elif answer_search3:
    answer = answer_search3.group(1)
  else:
    answer = UNKNOWN_ANSWER_DEFAULT_STRING
  return answer.replace("\n", "").replace("*", "")


def _strip_only(answer):
  return answer.strip()


def _clean_model_response(x):
  x = x.strip().split("\n")[-1]
  if "**Answer:**" in x:
    x = x.split("**Answer:**")[1]
  elif "Answer:" in x:
    x = x.split("Answer:")[1]

  x = x.replace("*", "").replace(".", "")
  x = x.strip()
  return x


def _parse_python_code_list_answer(llm_answer):
  """Parses llm_answer for python list."""
  # turn '```python' to '```'
  processed_answer = llm_answer.replace("```python", "```")
  code_block_start = processed_answer.find("```")
  if code_block_start == -1:
    return UNKNOWN_ANSWER_DEFAULT_STRING
  code_block_end = processed_answer.find("```", code_block_start + 3)
  if code_block_end == -1:
    return UNKNOWN_ANSWER_DEFAULT_STRING
  list_block_start = processed_answer.find(
      "[", code_block_start, code_block_end)
  if list_block_start == -1:
    return UNKNOWN_ANSWER_DEFAULT_STRING
  list_block_end = processed_answer.find(
      "]", list_block_start, code_block_end)
  if list_block_end == -1:
    return UNKNOWN_ANSWER_DEFAULT_STRING
  return processed_answer[list_block_start : list_block_end + 1]


def _syllogism_clean_model_response(answer):
  """Syllogism parsing implementation.

  Args:
    answer: The string of the raw LLM answer.

  Returns:
    The parsed answer.
  """
  out = answer.strip().split("\n")[-1]
  if "**Answer:**" in out:
    out = out.split("**Answer:**")[1]
  elif "Answer:" in out:
    out = out.split("Answer:")[1]

  out = out.replace("*", "").replace(".", "")
  out = out.strip()
  if not out:
    out = answer.strip()
    if "**Answer:**" in out:
      out = out.split("**Answer:**")[0]
    elif "Answer:" in out:
      out = out.split("Answer:")[0]

    out = out.replace("*", "").replace(".", "")
    out = out.strip()
  return out


PARSING_FN = immutabledict.immutabledict({
    ParsingFunctionEnum.STRIP_ONLY: _strip_only,
    ParsingFunctionEnum.REGEX_NAIVE1: _regex_naive1,
    ParsingFunctionEnum.CLEAN_MODEL_RESPONSE: _clean_model_response,
    ParsingFunctionEnum.BASE: lambda x: x,
    ParsingFunctionEnum.SYLLOGISM_CLEAN_MODEL_RESPONSE: (
        _syllogism_clean_model_response
    ),
    ParsingFunctionEnum.PARSE_PYTHON_CODE_LIST: _parse_python_code_list_answer,
})


# ==== Prommpting Types ====
ALL_STR_PROMPT_STRING = "ALL"


# If *_ALL is requested, then all prompts in the same StrEnum will be requested.
class CommonPromptEnum(enum.StrEnum):
  """Common prompts for all modules."""
  DEFAULT_ALL = ALL_STR_PROMPT_STRING
  PROMPT_1 = "{question}"
  PROMPT_2 = "{question}\nAnswer in only one word."
  PROMPT_3 = (
      "{question}\nThink through your answer then respond at the end "
      "with a newline and 'Answer:' with your answer. "
      "Use only one word for the answer."
  )
  PROMPT_3_MULTI = (
      "{question}\nThink through your answer then respond at the end "
      "with a newline and 'Answer:' with your answer."
  )
  PROMPT_4 = "{question}\nLet's think step by step"


class SocialNetworkPromptEnum(enum.StrEnum):
  SOCIAL_DEFAULT_ALL = ALL_STR_PROMPT_STRING
  SOCIAL_PROMPT_1 = (
      "You are a language model with advanced cognitive abilities. "
      "Your task is to understand and reason about the following "
      "social scenario, much like a human would. "
      "Read the story carefully and answer the questions that follow\n"
      "{question}"
  )


class ComparisonPromptEnum(enum.StrEnum):
  """Enum for comparison prompts."""

  COMPARISON_DEFAULT_ALL = ALL_STR_PROMPT_STRING
  COMPARISON_PROMPT_1 = (
      "{question}\nAnswer the above relational reasoning question with Yes or"
      " No. Use only one word for the answer."
  )
  COMPARISON_PROMPT_3 = (
      "You are a language model being probed for your reasoning abilities."
      " Your task is to carefully think about the following information"
      " and answer the question.\n{question}\n. Make sure to respond at the"
      " end with 'Answer:'"
  )
  COMPARISON_PROMPT_5 = (
      "{question}\nAnswer the above relational reasoning question with Yes or"
      " No with a newline and 'Answer:' with your answer. "
      "Give your best guess if uncertain. Use only one word for the answer."
  )
  COMPARISON_PROMPT_6 = (
      "{question}\nAnswer the above relational reasoning question with only "
      " Yes or No with a newline and 'Answer:'. "
      "Give your best guess if uncertain. Use only one word for the answer."
  )


class ComparisonIndeterminatePromptEnum(enum.StrEnum):
  """Comparison prompts that probe for uncertainty."""

  COMPARISON_INDETERMINATE_DEFAULT_ALL = ALL_STR_PROMPT_STRING
  COMPARISON_INDETERMINATE_1 = (
      "{question}\nAnswer the above relational reasoning question with Yes, No,"
      " or Unknown. Use Unknown if the question cannot be answered with "
      "the information given. Use only one word for the answer."
  )


class ComparisonConsistentPromptEnum(enum.StrEnum):
  """Comparison prompts that probe for detecting inconsistency."""

  COMPARISON_CONSISTENT_DEFAULT_ALL = ALL_STR_PROMPT_STRING
  COMPARISON_CONSISTENT_1 = ComparisonPromptEnum.COMPARISON_PROMPT_3.value


class SyllogismPrompt(enum.StrEnum):
  """Syllogism Prompts."""

  SYLLOGISM_DEFAULT_ALL = ALL_STR_PROMPT_STRING
  SYLLOGISM_1 = (
      "{question}\nThink through your answer then respond at the end "
      "with a newline and 'Answer:' with your answer."
  )


StrPrompt = enum.StrEnum


# ==== Generic Classes ====
class DatasetSplit(enum.StrEnum):
  TRAIN = "train"
  VAL = "val"
  TEST = "test"


@dataclasses.dataclass(frozen=True)
class Entity:
  """Represents an entity in any module/example.

  text: The text information of the Entity like name or type
  image: Any, not implemented yet.
  """

  text: str
  image: Any = dataclasses.field(hash=False)


@dataclasses.dataclass(frozen=True)
class CongruentObjectEntity(Entity):
  """An entity with a specific size and weight that can be compared.

  weight: How heavy an entity is in kilograms.
  size: How large an entity is in meters.
  """

  weight: float = dataclasses.field(default=-1.0, hash=False)
  size: float = dataclasses.field(default=-1.0, hash=False)


# Class for invalid data generation (usually generate an invalid graph)
class DataGenerationError(RuntimeError):
  """Exception for invalid data generation."""


# ==== Configs ====
class Ordering(enum.StrEnum):
  """Ordering for the entities in the prompt."""

  INORDER = "inorder"
  REVERSE = "reverse"
  RANDOM = "random"


class EntityMode(enum.StrEnum):
  """Mode for how to generate entities."""

  PRESET = "preset"
  INPUT = "input"
  CUSTOM = "custom"

  CONGRUENT = "congruent"
  INCONGRUENT = "incongruent"


# Comparison relation types and congruency modes.
class RelationType(enum.StrEnum):
  SIZE_NATURALTEXT = "size"
  WEIGHT_NATURALTEXT = "weight"
  AGE_NATURALTEXT = "age"


class CongruencyMode(enum.StrEnum):
  ALL_CONGRUENT = "all_congruent"  # all relationships will be congruent.
  ALL_INCONGRUENT = "all_incongruent"  # all relationships will be incongruent.
  RANDOM_NAME = "random_name"  #  object names will be randomly generated.
  RANDOM = "random"  # all objects will be shuffled to construct an order.


@dataclasses.dataclass
class ModelRunSpec:
  model_label: str = ""
  human_readable_label: str = ""
  model_class: Any = None
  num_workers: int = -1
  optimal_prompt: Any = None
  optimal_parse: Any = None
