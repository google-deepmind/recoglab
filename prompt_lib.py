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

"""Library code for accessing and manipulating prompting functions.

The goal is to make a library of prompts easily available for validation
purposes.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Set
import os
from typing import Any
from recoglab import common_types
from recoglab import eval_io_lib


def generate_modified_prompt_iterator(
    examples: Iterable[Mapping[str, Any]],
    prompt: common_types.StrPrompt,
    decode_tf: bool = False,
) -> Iterator[tuple[str, Mapping[str, Any]]]:
  """Processes each element with additional prompt formatting.

  Args:
    examples: Mappings of python primitives or tf primitives of evaluation
      examples.
    prompt: A StrPrompt enum to use.
    decode_tf: Whether to convert tf to python primitives.

  Yields:
    A tuple of the revised prompt and original dictionary.
  """
  for example in examples:
    if decode_tf:
      converted_example = eval_io_lib.tf_to_python_primitives(example)
    else:
      converted_example = example
    yield (prompt.value.format(**converted_example), converted_example)


def process_prompt_enum(
    requested_prompts: Iterable[common_types.StrPrompt],
) -> Set[common_types.StrPrompt]:
  """Given multiple StrPrompt, expands then compresses to minimal prompts.

  Args:
    requested_prompts: Prompts to test

  Returns:
    A set of prompts that expand *_ALL inputs and deduplicate prompts.
  """
  prompts_to_use = set()
  for prompt in requested_prompts:
    if prompt.value == common_types.ALL_STR_PROMPT_STRING:
      # Insert all prompts except the *_ALL one.
      prompts_to_use.update(
          member
          for member in type(prompt)
          if member.value != common_types.ALL_STR_PROMPT_STRING
      )
    else:
      prompts_to_use.add(prompt)
  return prompts_to_use


def _create_base_prediction(
    llm_base_prediction: str, example_dict: Mapping[str, Any]
) -> dict[str, str]:
  return {'index': str(example_dict['index']), 'pred': llm_base_prediction}


def _parse_prediction(
    base_prediction: Mapping[str, str], parse_fn: Callable[[str], str]
) -> Mapping[str, str]:
  """Parse Predictions."""
  if 'exception' in base_prediction:
    return base_prediction
  try:
    refined_prediction = parse_fn(base_prediction['pred'])
    return {'pred': refined_prediction, 'index': base_prediction['index']}
  except Exception as e:  # pylint: disable=broad-except
    return {
        'exception': repr(e),
        'pred': repr(e),
        'index': base_prediction['index'],
    }


class PromptSweeperHelper:
  """Helper class to manage IO when doing a validation sweep.

  When validating, tfrecord is expected to be relatively small as we will
  be running many different prompt templates on it. When testing on the dataset,
  use only a single prompt.

  Attributes:
    tfrecord_path: filepath to the validation tfrecord that is also used for
      writing evaluation and analysis.
    sweepable_prompts: A sequence of common_types.StrPrompt to indicate which
      prompts to evaluate.
  """

  def __init__(
      self,
      tfrecord_path: str,
      additional_label: str,
      requested_prompts: Iterable[common_types.StrPrompt],
      parsing_callables: (
          Iterable[common_types.ParsingFunctionEnum] | None
      ) = None,
      checkpoint_frequency: int = 250,
  ):
    """Helper class initialization.

    Args:
      tfrecord_path: the filepath to a tfrecord to evaluate
      additional_label: Additional label to insert in the eval label.
      requested_prompts: A sequence of common_types.StrPrompt to indicate which
        prompts to evaluate.
      parsing_callables: A sequence of labels to apply to base predictions and
        save.
      checkpoint_frequency: If greater than 0, will run an inference
        checkpointing system to save predictions every checkpoint_frequency
        examples.
    """
    if parsing_callables is None:
      parsing_callables = common_types.PARSING_FN.keys()
    self._parsing_callables = set(
        parsing_callable for parsing_callable in parsing_callables
    )
    self.tfrecord_path = tfrecord_path
    self._checkpoint_frequency = checkpoint_frequency
    self._label = additional_label
    self.sweepable_prompts = process_prompt_enum(requested_prompts)
    self._tfds = eval_io_lib.tf_record_dataset_iterator(self.tfrecord_path)

    self._indices = set(
        eval_io_lib.tf_to_python_primitives(tf_example).get('index')
        for tf_example in self._tfds
    )

  def get_missing_example_indices(self, eval_label: str) -> Set[int]:
    """Gets the diff of missing predictions between eval_label and the tfrecord.

    Args:
      eval_label: The eval label to check predictions for.

    Returns:
      A set of indices that are missing predictions.
    """
    prediction_folder = self.tfrecord_path.removesuffix('.tfrecord')
    prediction_folder = prediction_folder + '_predictions'
    prediction_json_name = os.path.join(prediction_folder, f'{eval_label}.json')
    return eval_io_lib.get_missing_predictions(
        self._indices, prediction_json_name
    )

  def _get_evaluation_label(
      self,
      model_label: str,
      prompt_label: common_types.StrPrompt,
      parse_version: str | None = None,
  ) -> str:
    eval_label = f'{self._label}_prompt:{prompt_label.name}_model:{model_label}'
    if parse_version:
      return f'{eval_label}_parse:{parse_version}'
    return eval_label

  def _get_analysis_label(self, model_label: str) -> str:
    return f'{self._label}_model:{model_label}'

  def _eval(
      self,
      llm_callable: Callable[[str, Mapping[str, Any]], str],
      prompt: common_types.StrPrompt,
      run_indices: Set[int] | None = None,
      max_number_examples: int = -1,
  ) -> list[dict[str, str]]:
    """Main thread processing of dataset ignoring ignored_indices.

    This does not ignore Exceptions and can be used for debugging.

    Args:
      llm_callable: A function that returns a prediction to a prompt.
      prompt: A common_types.StrPrompt with its value as the prompt template.
      run_indices: A set of indices to run on. If None, run on all.
      max_number_examples: If greater than 0, the number of examples to
        evaluate.

    Returns:
      A list of predictions.
    """
    if run_indices is None:
      run_indices_set = self._indices
    else:
      run_indices_set = run_indices
    need_to_run = lambda index: index in run_indices_set
    prompt_predictions = []
    if max_number_examples > 0:
      ds = self._tfds.take(max_number_examples)
    else:
      ds = self._tfds
    for example in generate_modified_prompt_iterator(
        ds, prompt, decode_tf=True
    ):
      new_question_with_template, py_dict = example
      if not need_to_run(py_dict['index']):
        continue
      prompt_predictions.append(
          _create_base_prediction(
              llm_callable(new_question_with_template, py_dict), py_dict
          )
      )
    return prompt_predictions

  def test_custom_prompt(
      self,
      llm_callable: Callable[[str, Mapping[str, Any]], str],
      model_label: str,
      prompt: common_types.StrPrompt,
  ) -> str:
    """Sets up a single evaluation loop with a custom StrPrompt.

    Note prompt_label will be prepended with CUSTOM_ to indicate that this
    was an unofficial prompt template. The caller is responsible for saving
    what that prompt was.

    Args:
      llm_callable: A function that returns a prediction to a prompt.
      model_label: An identifier for this model.
      prompt: A common_types.StrPrompt with its value as the prompt template.

    Returns:
      Filepath to the evaluation label.
    """
    eval_label = self._get_evaluation_label(model_label, prompt)
    prompt_predictions = self._eval(llm_callable, prompt, run_indices=None)
    return eval_io_lib.write_model_prediction_to_cns(
        self.tfrecord_path,
        eval_label,
        prompt_predictions,
        merge_predictions=False,
    )

  def _run_parse(
      self,
      base_predictions: Iterable[Mapping[str, str]],
      parse_fn: Callable[[str], str],
  ) -> list[Mapping[str, str]]:
    new_parsed_predictions = []
    for prompt_prediction in base_predictions:
      new_parsed_predictions.append(
          _parse_prediction(prompt_prediction, parse_fn)
      )
    return new_parsed_predictions

  def test_custom_parse(
      self,
      eval_fpath: str,
      custom_parsing_fn: Callable[[str], str],
      parsing_label: str,
  ) -> str:
    """Runs a custom parser on eval_fpath.

    eval_fpath should be the output ran only on the BASE parser which contains
    unparsed predictions.

    Args:
      eval_fpath: Path to BASE parser predictions
      custom_parsing_fn: custom parsing function.
      parsing_label: parsing label to use.

    Returns:
      Filepath to the evaluation label.
    """
    base_predictions = eval_io_lib.load_predictions(eval_fpath)
    new_parsed_predictions = self._run_parse(
        base_predictions, custom_parsing_fn
    )

    original_eval_label = os.path.basename(eval_fpath).removesuffix('.json')
    new_eval_label = f'{original_eval_label}_customparse:{parsing_label}'
    return eval_io_lib.write_model_prediction_to_cns(
        self.tfrecord_path,
        new_eval_label,
        new_parsed_predictions,
        merge_predictions=False,
    )

  def _sweep_parse(
      self,
      model_label: str,
      prompt: common_types.StrPrompt,
      base_predictions: Iterable[Mapping[str, str]],
  ) -> list[str]:
    """Sweeps base predictions over the parse functions.

    Args:
      model_label: The label of the base prediction.
      prompt: The prompt used with the model.
      base_predictions: Base predictions produced by the model.

    Returns:
      A list of evaluation paths for the predictions parsed
    """
    eval_fpath = []
    # Now write parsing predictions.
    for parsing_function in self._parsing_callables:
      parsed_label = self._get_evaluation_label(
          model_label, prompt, parsing_function.name
      )
      new_parsed_predictions = self._run_parse(
          base_predictions, common_types.PARSING_FN[parsing_function]
      )
      eval_fpath.append(
          eval_io_lib.write_model_prediction_to_cns(
              self.tfrecord_path,
              parsed_label,
              new_parsed_predictions,
              merge_predictions=False,
          )
      )
    return eval_fpath

  def sweep_and_write_evals(
      self,
      llm_callable: Callable[[str, Mapping[str, Any]], str],
      model_label: str,
      overwrite: bool = False,
      max_number_examples: int = -1,
      merge_predictions: bool = True,
  ) -> list[str]:
    """Evaluates llm_callable on the dataset with custom prompting.

    llm_callable accepts the whole dictionary containing other parts of the
    question and answer too. It's the function's responsibility to use
    the inputs fairly and parse their response for a prediction.

    Args:
      llm_callable: A function that returns a prediction to a prompt.
      model_label: An identifier for this model.
      overwrite: If true, will run every eval no matter what.
      max_number_examples: If greater than 0, the number of examples to
        evaluate.
      merge_predictions: If True, will skip old valid predictions and only run
        on invalid predictions before merging the two versions.

    Returns:
      A list to the predictions
    """
    eval_fpath = []
    for prompt in self.sweepable_prompts:
      label = self._get_evaluation_label(model_label, prompt)
      run_indices = None
      run_llm_evals = True
      if not overwrite and merge_predictions:
        # overwrite skips this check and runs on all indices.
        run_indices = self.get_missing_example_indices(label)
        if not run_indices:
          # Here run_indices is a set and so the implicit check is for
          # testing whether the set is empty.
          run_llm_evals = False
      if run_llm_evals:
        # Run analysis on the missing indices.
        prompt_predictions = self._eval(
            llm_callable,
            prompt,
            max_number_examples=max_number_examples,
            run_indices=run_indices,
        )
        base_eval_fpath = eval_io_lib.write_model_prediction_to_cns(
            self.tfrecord_path,
            label,
            prompt_predictions,
            merge_predictions=merge_predictions,
        )
      else:
        # Load base evaluation as it already exists.
        base_eval_fpath = eval_io_lib.get_prediction_path(
            self.tfrecord_path, label
        )
      prompt_predictions = eval_io_lib.load_predictions(base_eval_fpath)
      eval_fpath.append(base_eval_fpath)
      eval_fpath.extend(
          self._sweep_parse(
              model_label,
              prompt,
              prompt_predictions,
          )
      )
    return eval_fpath

  def aggregate_predictions(
      self,
      eval_fpaths: Iterable[str],
      model_label: str,
      overwrite: bool = False,
      max_number_examples: int | None = None,
  ) -> str:
    """Aggregates the predictions and writes them to disk.

    Args:
      eval_fpaths: a list of filepaths to different predictions
      model_label: An identifier for this model.
      overwrite: If true, will run even if evals already exist.
      max_number_examples: the number of examples to aggregate. Default is all.

    Returns:
      A filepath to the analysis json.
    """
    eval_labels = [
        os.path.basename(eval_fpath).removesuffix('.json')
        for eval_fpath in eval_fpaths
    ]
    return eval_io_lib.aggregate_model_predictions_to_record(
        self.tfrecord_path,
        self._get_analysis_label(model_label),
        eval_labels,
        overwrite=overwrite,
        n_examples=max_number_examples,
    )
