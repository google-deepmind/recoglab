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

"""Library for managing the eval of ReCogLab methods."""

import ast
from collections.abc import Callable, Collection, Mapping, Sequence, Set
import json
import os
# pickle is only used to save and load model run specs for internal use only.
from typing import Any

import immutabledict
import tensorflow as tf

from recoglab import recoglab_dataset


DEFAULT_MULTI_ANSWER_SEPARATOR = '\t'  # default answer splitter in strings
_FEATURE_SPEC = immutabledict.immutabledict({
    'index': tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    'question': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'question_only': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'answer': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'metadata': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'alternative_answers': tf.io.FixedLenFeature(
        [], tf.string, default_value=''
    ),
})


def parse_example(serialized_item: bytes) -> dict[str, Any]:
  """Parse a tf example for its content.

  Args:
    serialized_item: serialized example.

  Returns:
    unserialized tf.Example in a dictionary and tf.Tensor form.
  """
  return tf.io.parse_example(serialized_item, _FEATURE_SPEC)


def recoglab_dataset_example_to_tf_example(
    example: recoglab_dataset.ReCogLabDatasetExample,
    index: int,
    save_context_separately: bool = False,
) -> tf.train.Example:
  r"""Encodes ReCogLab example to a tf.Example.

  \t is the default character separator to split examples with multiple
  answers in string representation.

  Args:
    example: ReCogLab dataset example with prompts, questions, and answers.
    index: Used to identify the example within a dataset
    save_context_separately: If true, will save the context separately from the
      question.

  Returns:
    tf.train.Example encoded with features.
  """
  information = example.get_prompts()
  context_blocks = []
  for block in example.all_blocks:
    text = block.prompt_separator.join(block.prompt)
    context_blocks.append(text)
  questions = example.get_question()
  question_string = questions
  answers = example.get_answers()
  answer_string = answers[0]
  alternative_string = DEFAULT_MULTI_ANSWER_SEPARATOR.join(answers[1:])
  metadata = example.get_metadata()
  question_all = f'{information}\n{question_string}'
  # multiple answer blocks are separated by answer_block_separator
  # multiple answers in a single block are separated by multi_answer_separator
  if save_context_separately:
    feature = {
        'question': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[question_string.encode('utf-8')]
            )
        ),
        'context_blocks': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[block.encode('utf-8') for block in context_blocks]
            )
        ),
        'response': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[answer_string.encode('utf-8')])
        ),
        'other_responses': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[alternative_string.encode('utf-8')]
            )
        ),
        'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'metadata': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[repr(metadata).encode('utf-8')
                                                 ])
        ),
    }
  else:
    feature = {
        'question': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[question_all.encode('utf-8')])
        ),
        'question_only': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[question_string.encode('utf-8')]
            )
        ),
        'answer': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[answer_string.encode('utf-8')])
        ),
        'alternative_answers': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[alternative_string.encode('utf-8')]
            )
        ),
        'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
        'metadata': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[repr(metadata).encode('utf-8')]
            )
        ),
    }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def tf_record_dataset_iterator(filepath: str) -> tf.data.Dataset:
  """Returns an iterator over the dataset.

  Args:
    filepath: Filepath to the tfrecord

  Returns:
    An iterator that will pass through the dataset once.
  """
  dataset = tf.data.TFRecordDataset(filepath)
  return dataset.map(parse_example)


def get_missing_predictions(
    example_indices: Set[int],
    filepath_to_prediction: str,
) -> Set[int]:
  """Returns the indices that are missing predictions.

  Args:
    example_indices: The set of example_indies we want to run.
    filepath_to_prediction: the prediction_label to compare to search for
      missing predictions.
  """
  if not os.path.exists(filepath_to_prediction):
    return example_indices

  predictions = load_predictions(filepath_to_prediction)

  existing_predictions = {int(k['index']): k for k in predictions}

  need_to_rerun = set()
  for example_ind in example_indices:
    if example_ind not in existing_predictions:
      need_to_rerun.add(example_ind)
    else:
      if 'exception' in existing_predictions[example_ind]:
        need_to_rerun.add(example_ind)
      else:
        continue
  return need_to_rerun


def write_model_prediction_to_cns(
    tfrecord_source_filepath: str,
    prediction_set_label: str,
    predictions: Sequence[Mapping[str, str]],
    merge_predictions: bool = False,
) -> str:
  """Writes a json dump of the predictions to a folder of prediction_set_labels.

  When merging predictions, the rules are that the most recent 'valid'
  prediction between different versions is written. We always prefer valid
  predictions over invalid predictions. If both versions are invalid,
  then we prefer the most recent version.
  Validity is determined by whether the prediction has an 'exception' field.

  Args:
    tfrecord_source_filepath: the source of the tfrecord with eval data.
    prediction_set_label: the label for the prediction set
    predictions: A list of predictions. 'index' and 'pred' must be defined if
      joining with the original tf records.
    merge_predictions: If true, will attempt to merge with existing predictions,
      otherwise will overwrite the existing predictions.

  Returns:
    A filepath to the newly written eval results.
  """
  prediction_folder = tfrecord_source_filepath.removesuffix('.tfrecord')
  prediction_folder = prediction_folder + '_predictions'
  if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

  prediction_json_name = os.path.join(
      prediction_folder, f'{prediction_set_label}.json'
  )
  if merge_predictions and os.path.exists(prediction_json_name):
    # Attempt to merge predictions with existing predictions.
    existing_predictions = {
        existing_pred.get('index', '-1'): existing_pred
        for existing_pred in load_predictions(prediction_json_name)
    }
    for newest_prediction in predictions:
      prediction_ind = newest_prediction['index']
      existing_pred = existing_predictions.get(prediction_ind)
      if existing_pred is None:
        # No existing prediction found.
        existing_predictions[prediction_ind] = newest_prediction
      elif 'exception' in existing_pred:
        # Existing prediction is invalid so always write newest prediction.
        existing_predictions[prediction_ind] = newest_prediction
      elif 'exception' in newest_prediction:
        # Existing prediction is valid and newest prediction has an exception.
        continue
      else:
        # Both predictions are valid, so write the newest prediction.
        existing_predictions[prediction_ind] = newest_prediction
    prediction_to_write = list(existing_predictions.values())
  else:
    prediction_to_write = predictions
  # JSON dumped for ease, will be joined to the original file on demand.
  with open(prediction_json_name, 'w') as f:
    json.dump(prediction_to_write, f)
  return prediction_json_name


def get_prediction_path(
    tfrecord_source_filepath: str, prediction_set_label: str
) -> str:
  """Constructs a filepath to the prediction.

  Args:
    tfrecord_source_filepath: the source of the tfrecord with eval data.
    prediction_set_label: the label for the prediction set

  Returns:
    The save location of prediction_set_label on tfrecord_source_filepath.
  """
  prediction_folder = tfrecord_source_filepath.removesuffix('.tfrecord')
  prediction_folder = f'{prediction_folder}_predictions'
  return os.path.join(prediction_folder, f'{prediction_set_label}.json')


def is_existing_analysis(
    tfrecord_source_filepath: str, prediction_set_label: str
) -> bool:
  """Returns true if the evaluation has been run before.

  This check can be ignored if the evaluation must be rerun again, but exists
  just to check if an evaluation can be skipped.

  Args:
    tfrecord_source_filepath: the source of the tfrecord with eval data.
    prediction_set_label: the label for the prediction set

  Returns:
    True if a file exists with this tf_record and prediction_set_label.
  """
  return os.path.exists(
      get_prediction_path(tfrecord_source_filepath, prediction_set_label)
  )


def load_predictions(eval_fpath: str) -> list[dict[str, str]]:
  with open(eval_fpath, 'r') as f:
    return json.load(f)


def tf_to_python_primitives(
    dict_example: Mapping[str, tf.Tensor],
) -> Mapping[str, Any]:
  """Converts a dict of tf Tensors to python primitives.

  Args:
    dict_example: a dictionary outputted by a tf dataset for an example.

  Returns:
    A dictionary with values that are python primitives like strings, int, and
    dict.
  """
  output = {
      'index': int(dict_example['index'].numpy()),
      'question': dict_example['question'].numpy().decode(),
      'answer': dict_example['answer'].numpy().decode(),
  }
  alternative_answers_str = dict_example['alternative_answers'].numpy().decode()
  if alternative_answers_str:
    output['alternative_answers'] = alternative_answers_str.split(
        DEFAULT_MULTI_ANSWER_SEPARATOR
    )
  else:
    output['alternative_answers'] = []
  metadata_repr = dict_example['metadata'].numpy().decode()
  if metadata_repr:
    output['metadata'] = ast.literal_eval(metadata_repr)
  else:
    output['metadata'] = {}
  return output


def aggregate_model_predictions_to_record(
    tfrecord_source_filepath: str,
    eval_label: str,
    prediction_labels_to_save: Sequence[str],
    overwrite: bool = False,
    n_examples: int | None = None,
) -> str:
  """Aggregates prediction_set_label to a common file.

  dictionary
    "index" ->
        dictionary
          "metadata" -> dict[str, str]
          "answers" -> list[str]
          "predictions/{prediction_label}" -> str

  Args:
    tfrecord_source_filepath: the source of the tfrecord with eval data.
    eval_label: the label to apply on the joined predictions
    prediction_labels_to_save: the prediction_labels to use.
    overwrite: If false, will raise an exception if overwriting an existing set
      of predictions.
    n_examples: Number of examples to record, if -1 will record all indices.

  Raises:
    FileExistsError: if overwrite is disabled and a joined prediction set
      already exists.
    FileNotFoundError: if a non-existent prediction label set is requested.
    KeyError: if a prediction is missing 'index' or 'pred'

  Returns:
    A string to the aggregated eval results that can be used for analysis.
  """
  tfrecord = tfrecord_source_filepath.removesuffix('.tfrecord')
  eval_folder = f'{tfrecord}_eval'
  eval_json_name = os.path.join(eval_folder, f'{eval_label}.json')
  if not overwrite and os.path.exists(eval_json_name):
    raise FileExistsError(
        f'Eval label {eval_label} already exists for: {tfrecord}'
    )

  prediction_folder = f'{tfrecord}_predictions'
  # Fail fast if any of the predictions are missing.
  for prediction_labels in prediction_labels_to_save:
    prediction_json_name = os.path.join(
        prediction_folder, f'{prediction_labels}.json'
    )
    if not os.path.exists(prediction_json_name):
      raise FileNotFoundError(
          f'Requesting non-existent eval runs: {prediction_labels}'
      )
  dataset = tf_record_dataset_iterator(tfrecord_source_filepath)
  all_metrics_for_analysis = {}
  for dict_example in dataset:
    py_dict_example = tf_to_python_primitives(dict_example)
    index = py_dict_example['index']
    if n_examples and index >= n_examples: continue
    answers = [py_dict_example['answer']]
    answers.extend(py_dict_example['alternative_answers'])
    all_metrics_for_analysis[index] = {
        'answers': answers,
        'metadata': py_dict_example['metadata'],
    }

  for prediction_labels in prediction_labels_to_save:
    prediction_json_name = os.path.join(
        prediction_folder, f'{prediction_labels}.json'
    )
    inference_runs = load_predictions(prediction_json_name)
    for inference in inference_runs:
      if 'index' not in inference or 'pred' not in inference:
        raise KeyError(
            'index or pred were not found in '
            f'the prediction set {prediction_labels}'
        )
      if n_examples and int(inference['index']) >= n_examples:
        continue
      if 'exception' in inference:
        all_metrics_for_analysis[int(inference['index'])][
            f'exception/{prediction_labels}'
        ] = inference['exception']
      else:
        all_metrics_for_analysis[int(inference['index'])][
            f'predictions/{prediction_labels}'
        ] = inference['pred']
  if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)
  with open(eval_json_name, 'w') as f:
    json.dump(all_metrics_for_analysis, f)
  return eval_json_name


def include_string_match_metadata(
    metadata_field: str, valid_values: Collection[str]
) -> Callable[[dict[str, str]], bool]:
  """Produces a function that checks whether dictionary passes a string match.

  True iff metadata[metadata_field] in valid_values

  Args:
    metadata_field: The name of the metadata field to test
    valid_values: The set of string values to include analysis.

  Returns:
    A function that checks metadata_dict for string equality.
  """

  def _check(metadata_dict):
    try:
      return metadata_dict[metadata_field] in valid_values
    except KeyError:
      return False
  return _check


def include_greater_equal_metadata(
    metadata_field: str, lower_bound: float
) -> Callable[[dict[str, str]], bool]:
  """Produces a function that checks whether dictionary passes a string match.

  True iff metadata[metadata_field] >= lower_bound

  Args:
    metadata_field: The name of the metadata field to test
    lower_bound: The lowest acceptable value

  Returns:
    A function that checks metadata_dict for >=.
  """

  def _check(metadata_dict):
    try:
      return float(metadata_dict[metadata_field]) >= lower_bound
    except KeyError:
      return False
  return _check


def include_lesser_equal_metadata(
    metadata_field: str, upper_bound: float
) -> Callable[[dict[str, str]], bool]:
  """Produces a function that checks whether dictionary passes a string match.

  True iff metadata[metadata_field] <= upper_bound

  Args:
    metadata_field: The name of the metadata field to test
    upper_bound: The highest acceptable value

  Returns:
    A function that checks metadata_dict for <=.
  """

  def _check(metadata_dict):
    try:
      return float(metadata_dict[metadata_field]) <= upper_bound
    except KeyError:
      return False
  return _check


def convert_input_condition_to_testable_condition(
    meta_data_analysis_conditions: Sequence[str],
) -> Callable[[dict[str, str]], bool]:
  """Turns a list of conditions to a function that checks all conditions pass.

  See `score_evaluation_runs` documentation for notation.

  Args:
    meta_data_analysis_conditions: A list of clauses to filter out examples.

  Raises:
    ValueError: If any condition in meta_data_analysis_conditions is malformed.

  Returns:
    True if all conditions pass.
  """
  # Empty conditions defaults to always returning True.
  testable_fns = [lambda _: True]
  # Construct a function for condition
  for condition in meta_data_analysis_conditions:
    if '==' in condition:
      field_name, valid_values_str = condition.split('==')
      valid_values = valid_values_str.split(',')
      testable_fns.append(
          include_string_match_metadata(field_name, valid_values)
      )
    elif '>=' in condition:
      field_name, lower_bound_str = condition.split('>=')
      lower_bound = float(lower_bound_str)
      testable_fns.append(
          include_greater_equal_metadata(field_name, lower_bound)
      )
    elif '<=' in condition:
      field_name, upper_bound_str = condition.split('<=')
      upper_bound = float(upper_bound_str)
      testable_fns.append(
          include_lesser_equal_metadata(field_name, upper_bound)
      )
    else:
      raise ValueError(f'Condition clause malformed: {condition}')

  return lambda item: all(fn(item) for fn in testable_fns)


def convert_code_list_answer_to_python_tuple(
    answer: str,
) -> Sequence[Any] | None:
  """Safely converts python code of a list answer to a python tuple.

  Args:
    answer: The answer to convert.

  Returns:
    A list of python primitives evaluated from statement or None if
    the answer couldn't be converted.
  """
  try:
    # strips leading white space in answer which breaks literal_eval.
    # trailing white space doesn't matter.
    answer = tuple(ast.literal_eval(answer.strip()))
    # if answer is unhashable then a TypeError is raised. Covers the case
    # where answer is a nested list.
    hash(answer)
    return answer
  except (ValueError, SyntaxError, TypeError):
    return None


def score_exact_path_predictions(
    labels: Sequence[str], prediction: str
) -> float:
  """Scores the output for exact match to the answer.

  Args:
    labels: The output of the converted code or None if it failed to decode an
      answer.
    prediction: The answer to compare the prediction to.

  Returns:
    A float between 0 and 1 representing the similarity of the prediction to the
    answer.
  """
  decoded_answers = {
      convert_code_list_answer_to_python_tuple(l) for l in labels
  }
  decoded_prediction = convert_code_list_answer_to_python_tuple(prediction)
  if decoded_prediction is None:
    return 0.0
  return float(decoded_prediction in decoded_answers)


def score_evaluation_runs(
    agg_prediction_filepath: str,
    meta_data_analysis_conditions: (
        Sequence[str] | Callable[[dict[str, str]], bool]
    ) = (),
    score_fn: Callable[[list[str], str], float] | None = None,
) -> dict[str, dict[str, Any]]:
  """Score aggregated predictions with filtering on metadata.

  Empty meta_data_analysis_conditions will evaluate on every example in
  agg_prediction_filepath.

  If an example is missing a metadata field, then it's automatically
  excluded from analysis. Otherwise we check the metadata field and
  test whether its value is permissible.

  An example must fulfill all conditions to be included in analysis.

  Notation:
    # Only analyze examples where metadata['key'] exactly matches '1' or 'a'
    # Checks for string equality only.
    meta_data_analysis_conditions = ['key==1,a']

    # Only analyze examples where metadata['key'] is between [3, 30]
    # Converts the metadata field to a float and runs the comparison.
    meta_data_analysis_conditions = ['key>=3', 'key<=30']

  Args:
    agg_prediction_filepath: The aggregated results to analyze
    meta_data_analysis_conditions: a list of clauses to filter out examples or a
      Callable that will return true or false given a metadata dictionary.
    score_fn: A function to score an example's prediction to a list of possible
      answers. Defaults to checking the pred is in answers.

  Returns:
    A dictionary of aggregated statistics for each model
  """
  if not isinstance(meta_data_analysis_conditions, Callable):
    include_fn = convert_input_condition_to_testable_condition(
        meta_data_analysis_conditions
    )
  else:
    include_fn = meta_data_analysis_conditions

  if not score_fn:
    # Default
    score_fn = lambda labels, pred: float(pred in labels)

  with open(agg_prediction_filepath, 'r') as f:
    agg_predictions = json.load(f)
  model_predictions = {}
  for _, prediction_outputs in agg_predictions.items():
    # we no longer care about the index since we're aggregating.
    valid = include_fn(prediction_outputs['metadata'])
    if not valid:
      continue
    for prediction_key in prediction_outputs.keys():
      if 'predictions/' not in prediction_key:
        continue
      else:
        # prediction/<eval_inference_label>
        eval_inference_label = prediction_key[12:]
      if eval_inference_label not in model_predictions:
        model_predictions[eval_inference_label] = {
            'total_scores': 0.0,
            'total_examples': 0,
            'variance': 0.0,
        }
      score = score_fn(
          prediction_outputs['answers'], prediction_outputs[prediction_key]
      )
      model_predictions[eval_inference_label]['total_scores'] += score
      model_predictions[eval_inference_label]['total_examples'] += 1
      model_predictions[eval_inference_label]['variance'] += score ** 2
  for model_name in model_predictions:
    model_stats = model_predictions[model_name]
    model_predictions[model_name]['variance'] = (
        model_stats['variance'] / model_stats['total_examples'] -
        (model_stats['total_scores'] / model_stats['total_examples']) ** 2
    )
  return model_predictions
