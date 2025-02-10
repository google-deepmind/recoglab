#!/bin/bash
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

base_path='/tmp/recoglab_dataset'



# num_examples=50
# # For seeded experiments:
# for seed in 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75; do
#   for task in social_network_FastestMessage_ExactPath social_network_FastestMessage_NumHops; do
#     for difficulty in basic hard; do
#       for i in 2 3; do
#         config_str="${task}_${difficulty}${i}"
#         python -m recoglab.generate_static_dataset \
#           --recoglab_configuration_str="$config_str" \
#           --split=val \
#           --output_path="${base_path}/reseeded/${seed}" \
#           --num_examples="$num_examples" \
#           --seed="$seed"
#       done
#     done
#   done
# done


# Flowery-noflowery language is flavor-text no-flavor-text from the paper.
experiment_name="flower_noflower"
PROD_SEED=42

# We only validated on the 20/30 entity split for each task respectively.
split='val'
num_examples=50
for task in social_network_FastestMessage_ExactPath social_network_FastestMessage_NumHops; do
  for difficulty in flower noflower; do
    for i in 20 30; do
      config_str="${task}_${difficulty}${i}"
      python -m recoglab.generate_static_dataset \
        --recoglab_configuration_str="$config_str" \
        --split="$split" \
        --output_path="${base_path}/${experiment_name}/${split}" \
        --num_examples="$num_examples" \
        --seed="$PROD_SEED"
    done
  done
done

# Test set 10 20 30 40 examples were evaluated on all models.
split='test'
num_examples=250
for task in social_network_FastestMessage_ExactPath social_network_FastestMessage_NumHops; do
  for difficulty in flower noflower; do
    for i in 10 20 30 40; do
      config_str="${task}_${difficulty}${i}"
      python -m recoglab.generate_static_dataset \
        --recoglab_configuration_str="$config_str" \
        --split="$split" \
        --output_path="${base_path}/${experiment_name}/${split}" \
        --num_examples="$num_examples" \
        --seed="$PROD_SEED"
    done
  done
done

# For some more advance models we generated 50-70 entities with only 100 test examples.
num_examples=100
for task in social_network_FastestMessage_ExactPath social_network_FastestMessage_NumHops; do
  for difficulty in flower noflower; do
    for i in 50 60 70; do
      config_str="${task}_${difficulty}${i}"
      python -m recoglab.generate_static_dataset \
        --eeaao_configuration_str="$config_str" \
        --split="$split" \
        --output_path="${base_path}/${experiment_name}/${split}" \
        --num_examples="$num_examples" \
        --seed="$PROD_SEED"
    done
  done
done

