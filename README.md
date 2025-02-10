<!-- mdlint off(LINE_OVER_80) -->
<!-- mdlint off(SNIPPET_INVALID_LANGUAGE) -->
# ReCogLab
[Project Page Coming Soon] |
[Paper Coming Soon]

This codebase accompanies the paper

**ReCogLab: A Framework
Testing Relational Reasoning & Cognitive Hypotheses on LLM** \
Andrew Liu, Others, Kenneth Marino

*Note that this is not an officially supported Google product.*

## About
This framework is designed to allow researchers in cognitive science and NLP to
quickly prototype language model experiments on a wide variety of relational
reasoning. Our framework automatically generates relational reasoning word
problems that can be used to probe for cognitive effects in Large Language
Models (LLMs). We also use this framework to identify problem settings that
negatively impact reasoning capabilities.

## Colab Demo
Coming Soon.

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

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

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
