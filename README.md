<!-- mdlint off(LINE_OVER_80) -->
<!-- mdlint off(SNIPPET_INVALID_LANGUAGE) -->
# ReCogLab
[Paper Coming Soon] | [[Colab]](https://colab.sandbox.google.com/github/google-deepmind/recoglab/blob/main/colab/generate_dataset_v1.ipynb)

This codebase accompanies the paper:

**ReCogLab: A Framework
Testing Relational Reasoning & Cognitive Hypotheses on LLM** \
Andrew Liu, Henry Prior, Gargi Balasubramaniam, Rivka Moroshko, Amir Zait, Ilia Labzovsky, Danny Karmon, Ishita Dasgupta, Kim Stachenfeld, Kenneth Marino

## About
ReCogLab is a generative framework designed to allow researchers in cognitive science and NLP to
quickly prototype language model experiments on a wide variety of relational
reasoning. Our framework automatically generates relational reasoning word
problems that can be used to probe for cognitive effects and diagnosing reasoning capabilities in Large Language
Models (LLMs). Our framework can generate more challenging examples along multiple
dimensions such as the number of entities, length of the chain, use of filler
and flavor text.

Several transitive inference capabilities of tasks that we can generate configurations for include:

* **Basic Transitive Inference**: Generate problems that evaluate basic transitive inference reasoning skills.
```
# Question
Dog is bigger than Apple.
Dog is smaller than Fire truck.
Is apple smaller than Fire truck?
# Answer
No
```
* **Graph Traversal**: Generate problems that evaluate multi-hop reasoning skills.
```
# Question
Anna is friends with Bob.
Bob is friends with Carl.
Bob is friends with Dana.
If a messsage can be exchanged between people who are friends, if Dana wants to pass a message to Anna, what's the exact path of friends that will receive the message?
# Answer
['Dana', 'Bob', 'Anna']
```
* **Consistency Detection**: Generate problems that evalute detecting locigally inconsistent statements.
```
# Question
Dog is bigger than Apple.
Dog is smaller than Fire truck.
Fire truck is smaller than Apple.
Are the above statements consistent or inconsistent with each other?
# Answer
Inconsistent
```
* **Determinacy / Indeterminacy / Feasibility**: Generates problems and ask whether sufficient information has been provided to answer the question.
```
# Question
Dog is bigger than Apple.
Dog is smaller than Fire truck.
Dog is smaller than Van.
Is Van bigger than Fire truck?
# Answer
Unknown
```
## Colab Demo
We have a [Colab demo](https://colab.sandbox.google.com/github/google-deepmind/recoglab/blob/main/colab/generate_dataset_v1.ipynb) that contains configurable form to
generate common examples with.

## Local Installation and Usage

We recommend using `virtualenv` to manage ReCogLab's dependency. See instructions for [installing virtualenv](https://virtualenv.pypa.io/en/latest/installation.html). For ICLR 2025, we use `jax==0.4.33`, `jaxlib==0.4.33`. Jax PRNGKeys are not deterministic across versions so to reproduce our experimental results exactly, one must use v0.4.33 JAX.

```bash
git clone https://github.com/google-deepmind/recoglab.git
virtualenv recoglab_venv

source recoglab_venv/bin/activate  # enters the recoglab virtualenv system
pip install -r requirements.txt  # installs the necessary packages to recoglab_venv
```

Now we can call `python -m recoglab.generate_static_dataset` to call the predefined
ReCogLab binary. This will write the `.tfrecord` of examples and `.config` to `output_path`.

```bash
python -m recoglab.generate_static_dataset \
  --recoglab_configuration_str="feasibile_infeasible_tree" \
  --split="test" --output_path="/tmp/test" --num_examples=50 --seed="42"
```

`recoglab_configuration_str` are predefined configurations that we used for basic Social Network, Comparison, Syllogism, and Family JSON experiments.
This can be customized or one can implement your own module of relational reasoning. Statement can be overridden through the command line argument

```bash
python -m recoglab.generate_static_dataset \
  --recoglab_configuration_str="feasibile_infeasible_tree" \
  --split="test" --output_path="/tmp/test" --num_examples=50 --seed="42" \
  --config_overwrite="num_entities_gen=10"
```

We also provide `iclr_dataset.sh` to reproduce the datasets that we generated for experimental results used in our paper.

## Citing this work

```bibtex
@inproceedings{
  liu2025recoglab,
  title={ReCogLab: a framework testing relational reasoning, cognitive hypotheses on {LLM}s},
  author={Andrew Liu and Henry Prior and Gargi Balasubramaniam and
        Rivka Moroshko and Amir Zait and Ilia Labzovsky and Danny Karmon and
        Ishita Dasgupta and Kim Stachenfeld and Kenneth Marino},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=yORSk4Ycsa}
  }
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
