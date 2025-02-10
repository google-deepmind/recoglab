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

"""Code for generating Family ReCogLab dataset."""

import enum
import json
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections

from recoglab import common_types
from recoglab.modules import recoglab_base

_HOBBIES = (
    "reading",
    "music",
    "cooking",
    "traveling",
    "cycling",
    "running",
    "painting",
    "writing",
    "gardening",
    "knitting",
    "dancing",
)


class RelationType(enum.StrEnum):
  """Different question types for family data."""

  # Compute the size of a said family
  FAMILY_SIZE = "family_size"

  # Identify if the hobby is a hobby of a family member
  FAMILY_MEMBER_HOBBY = "family_member_hobby"

  # Compare the size of two families
  FAMILY_SIZE_COMPARISON = "family_size_comparison"

  # Compare the age of two family members
  FAMILY_MEMBER_AGE_COMPARISON = "family_member_age_comparison"

  # Compare the hobby of two family members and find common hobbies
  FAMILY_MEMBER_HOBBY_COMPARISON = "family_member_hobby_comparison"


class FamilyMemberEntity(common_types.Entity):
  """Data class for a family member."""

  def __init__(
      self,
      text: str,
      age: Any,
      hobbies: list[str],
  ):
    super().__init__(text, None)
    self.age = age
    self.hobbies = hobbies

  def __eq__(self, other):
    if not isinstance(other, FamilyMemberEntity):
      return False
    # Compare relevant attributes for equality
    return (
        self.text == other.text
        and self.age == other.age
        and set(self.hobbies) == set(other.hobbies)
    )

  def __hash__(self):
    # Create a hash based on the same attributes used in __eq__
    return hash((self.text, self.age, self.hobbies))


class FamilyEntity(common_types.Entity):
  """Data class for a family."""

  def __init__(
      self,
      text: str,
      members: list[FamilyMemberEntity],
      address: str,
  ):
    super().__init__(text, None)
    self.members = members
    self.address = address

  def __eq__(self, other):
    if not isinstance(other, FamilyEntity):
      return False
    # Compare relevant attributes for equality
    return self.text == other.text and self.address == other.address

  def __hash__(self):
    # Create a hash based on the same attributes used in __eq__
    return hash((self.text, self.address))


def _pretty_print_family(family: FamilyEntity) -> str:
  """Generates a string of a nested json representation of a FamilyEntity."""

  family_data: dict[str, Any] = {
      "Family Name": family.text,
      "Address": family.address,
      "Members": [],
  }
  for member in family.members:
    member_data = {
        "Name": member.text,
        "Age": member.age.item(),
        "Hobbies": member.hobbies if member.hobbies else [],
    }
    family_data["Members"].append(member_data)  # pytype: disable=attribute-error

  json_str = json.dumps(
      family_data, indent=2
  )
  return json_str


class GenerateFamilyData:
  """Generates family data."""

  def __init__(
      self,
      cfg: ml_collections.ConfigDict,
      split: common_types.DatasetSplit,
  ):
    self.cfg = cfg
    self._first_names = {
        common_types.DatasetSplit.TEST: [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Emily",
            "Frank",
            "Grace",
            "Henry",
            "Ivy",
            "Jack",
            "Aisha",
            "Diego",
            "Sofia",
            "Kai",
            "Maya",
            "Liam",
            "Ava",
            "Noah",
            "Isabella",
            "Muhammad",
        ],
        common_types.DatasetSplit.VAL: [
            "Katie",
            "Liam",
            "Sofia",
            "Mateo",
            "Olivia",
            "Ethan",
            "Ava",
            "Noah",
            "Isabella",
            "Lucas",
            "Chloe",
            "Benjamin",
            "Mia",
            "Elijah",
            "Amelia",
        ],
        common_types.DatasetSplit.TRAIN: [
            "Ethan",
            "Ava",
            "Noah",
            "Isabella",
            "Lucas",
            "Mia",
            "Aiden",
            "Harper",
            "Grayson",
            "Evelyn",
            "Abigail",
            "Michael",
            "Sofia",
            "Daniel",
            "Madison",
        ],
    }
    self._last_names = {
        common_types.DatasetSplit.TEST: [
            "Garcia",
            "Lopez",
            "Gonzalez",
            "Chen",
            "Lee",
            "Rodriguez",
            "Williams",
            "Brown",
            "Davis",
            "Miller",
            "Wilson",
            "Moore",
            "Taylor",
            "Anderson",
            "Thomas",
        ],
        common_types.DatasetSplit.VAL: [
            "Kim",
            "Nguyen",
            "Tran",
            "Singh",
            "Patel",
            "Jackson",
            "White",
            "Harris",
            "Martin",
            "Thompson",
            "Robinson",
            "Clark",
            "Lewis",
            "Walker",
            "Hall",
        ],
        common_types.DatasetSplit.TRAIN: [
            "Shah",
            "Wong",
            "Liu",
            "Williams",
            "Jones",
            "Garcia",
            "Davis",
            "Rodriguez",
            "Martinez",
            "Hernandez",
            "Lopez",
            "Gonzalez",
            "Perez",
            "Sanchez",
            "Rivera",
        ],
    }
    self._streets = {
        common_types.DatasetSplit.TEST: [
            "Mission",
            "Market",
            "Castro",
            "Haight",
            "Fillmore",
            "Broadway",
            "Sunset",
            "Divisadero",
            "Geary",
            "Lombard",
            "Pine",
            "Bush",
            "Sutter",
            "Post",
        ],
        common_types.DatasetSplit.VAL: [
            "Geary",
            "Lombard",
            "Divisadero",
            "Valencia",
            "Telegraph",
            "Ashby",
            "College",
            "Adeline",
            "San Pablo",
            "MacArthur",
            "Fruitvale",
            "International",
            "Grand",
            "Lakeshore",
        ],
        common_types.DatasetSplit.TRAIN: [
            "Shattuck",
            "University",
            "El Camino Real",
            "Sand Hill Road",
            "Willow",
            "Charleston",
            "Meadow",
            "Oak",
            "Hillcrest",
            "Park",
            "Center",
            "Main",
            "First",
            "Second",
        ],
    }
    self._cities = {
        common_types.DatasetSplit.TEST: [
            "San Francisco",
            "Oakland",
            "San Jose",
            "Berkeley",
            "Palo Alto",
            "Fremont",
            "San Mateo",
            "Daly City",
            "Richmond",
            "Hayward",
        ],
        common_types.DatasetSplit.VAL: [
            "Mountain View",
            "Sunnyvale",
            "Fremont",
            "Santa Clara",
            "Richmond",
            "Berkeley",
            "San Leandro",
            "Union City",
            "Newark",
            "Milpitas",
        ],
        common_types.DatasetSplit.TRAIN: [
            "Hayward",
            "Concord",
            "Walnut Creek",
            "San Mateo",
            "Redwood City",
            "San Rafael",
            "Novato",
            "Vallejo",
            "Fairfield",
            "Antioch",
        ],
    }
    self.states = ["CA"]
    self.split = split
    self.hobby_images = {
        "reading": [
            "/x20/users/ga/gargisb/pictures/reading-1.jpg",
            "/x20/users/ga/gargisb/pictures/reading-2.jpeg",
        ],
        "music": [
            "/x20/users/ga/gargisb/pictures/music-1.jpg",
            "/x20/users/ga/gargisb/pictures/music-2.jpg",
        ],
        "cooking": [
            "/x20/users/ga/gargisb/pictures/cooking-1.jpg",
            "/x20/users/ga/gargisb/pictures/cooking-2.jpg",
        ],
        "traveling": [
            "/x20/users/ga/gargisb/pictures/travelling-1.jpg",
            "/x20/users/ga/gargisb/pictures/travelling-2.jpg",
        ],
        "cycling": [],
        "running": [],
        "painting": [],
        "writing": [],
        "gardening": [],
        "knitting": [],
        "dancing": [],
    }

  @property
  def first_names(self) -> list[str]:
    return self._first_names[self.split]

  @property
  def last_names(self) -> list[str]:
    return self._last_names[self.split]

  @property
  def streets(self) -> list[str]:
    return self._streets[self.split]

  @property
  def cities(self) -> list[str]:
    return self._cities[self.split]

  def generate_family(self, prng_key: jax.Array) -> FamilyEntity:
    """Generates a single family with members."""
    # Use jax.random for everything
    prng_key, *subkeys = jax.random.split(
        prng_key, 10
    )  # Generate enough subkeys
    family_size = jax.random.randint(
        subkeys[0], (), 1, self.cfg.max_members + 1
    )
    family_name = self.last_names[
        int(jax.random.randint(subkeys[1], (), 0, len(self.last_names)))
    ]
    address = (
        f"{int(jax.random.randint(subkeys[2], (), 1, 999))} {self.streets[int(jax.random.randint(subkeys[3], (), 0, len(self.streets)))]} St,"
        f" {self.cities[int(jax.random.randint(subkeys[4], (), 0, len(self.cities)))]},"
        f" {self.states[int(jax.random.randint(subkeys[5], (), 0, len(self.states)))]} {int(jax.random.randint(subkeys[6], (), 10000, 99999))}"
    )

    members = []
    for _ in range(family_size):
      prng_key, *member_subkeys = jax.random.split(prng_key, 5)
      age = jax.random.randint(member_subkeys[0], (), 1, 101)

      num_hobbies = jax.random.randint(
          member_subkeys[1], (), 1, len(_HOBBIES)
      ).item()
      hobby_indices = jax.random.choice(
          member_subkeys[2],
          jnp.arange(len(list(self.hobby_images.keys()))),
          (num_hobbies,),
          replace=False,
      )
      hobbies = [list(self.hobby_images.keys())[i] for i in hobby_indices]

      member = FamilyMemberEntity(
          text=self.first_names[
              int(
                  jax.random.randint(
                      member_subkeys[3], (), 0, len(self.first_names)
                  )
              )
          ],
          age=age,
          hobbies=hobbies,
      )
      members.append(member)

    family = FamilyEntity(
        text=family_name,
        members=members,
        address=address,
    )
    return family

  def generate_dataset(
      self, prng_key: jax.Array, num_families: int
  ) -> list[FamilyEntity]:
    """Generates the complete family dataset."""

    families: set[FamilyEntity] = set()  # Use a set for uniqueness
    while len(families) < num_families:
      prng_key, subkey = jax.random.split(prng_key)
      new_family = self.generate_family(subkey)
      # Add family to the set; it will only be added if it's unique
      # print("generated family: ")
      families.add(new_family)
      # print(len(families), num_families)

    return list(families)  # Convert back to a list


class FamilyQuestionGenerator:
  """Generates questions about family data."""

  def __init__(self, cfg: ml_collections.ConfigDict):
    self.cfg = cfg
    self.relation_generators = {
        RelationType.FAMILY_SIZE: self._generate_family_size_question,
        RelationType.FAMILY_MEMBER_HOBBY: (
            self._generate_family_member_hobby_question
        ),
        RelationType.FAMILY_SIZE_COMPARISON: (
            self._generate_family_size_comparison_question
        ),
        RelationType.FAMILY_MEMBER_AGE_COMPARISON: (
            self._generate_family_member_age_comparison_question
        ),
        RelationType.FAMILY_MEMBER_HOBBY_COMPARISON: (
            self._generate_family_member_hobby_comparison_question
        ),
    }

  def generate(
      self,
      prng_key: jax.Array,
      all_families: list[FamilyEntity],
  ) -> recoglab_base.ReCogLabQuestionAnswerBlock:
    """Generates a question block."""

    relation_type = self.cfg.relation_type
    generate_fn = self.relation_generators.get(relation_type)
    if not generate_fn:
      raise NotImplementedError(
          f"Question type {relation_type} not implemented."
      )
    return generate_fn(prng_key, all_families)

  def _generate_family_size_question(
      self,
      prng_key: jax.Array,
      all_families: list[FamilyEntity],
  ) -> recoglab_base.ReCogLabQuestionAnswerBlock:
    """Generates a family size question."""

    _, subkey = jax.random.split(prng_key, 2)

    # Initialize a QA block
    block = recoglab_base.ReCogLabQuestionAnswerBlock()

    # Choose family 1
    family1_index = int(jax.random.randint(subkey, (), 0, len(all_families)))
    family1 = all_families[family1_index]

    # Add address to diambiguiate in case last name clashes
    block.prompt.append(
        f"How many members are in the {family1.text} family living on"
        f" {family1.address}? Answer as a single number."
    )
    block.answers = [str(len(family1.members))]
    return block

  def _generate_family_member_hobby_question(
      self,
      prng_key: jax.Array,
      all_families: list[FamilyEntity],
  ) -> recoglab_base.ReCogLabQuestionAnswerBlock:
    """Generates a hobby identification question."""

    _, subkey1, subkey2, subkey3 = jax.random.split(prng_key, 4)

    # Initialize a QA block
    block = recoglab_base.ReCogLabQuestionAnswerBlock()

    # Choose family 1
    family1_index = int(jax.random.randint(subkey1, (), 0, len(all_families)))
    family1 = all_families[family1_index]

    # Choose member 1
    member1 = family1.members[
        int(jax.random.randint(subkey2, (), 0, len(family1.members)))
    ]

    # Choose a hobby
    hobby_to_ask = _HOBBIES[
        int(jax.random.randint(subkey3, (), 0, len(_HOBBIES)))
    ]

    is_hobby = hobby_to_ask in member1.hobbies

    # Add address to diambiguiate in case last name clashes
    block.prompt.append(
        f"Is {hobby_to_ask} a hobby of {member1.text} from the"
        f" {family1.text} family living on {family1.address}? Answer with Yes"
        " or No."
    )
    block.answers = ["Yes" if is_hobby else "No"]
    return block

  def _generate_family_size_comparison_question(
      self,
      prng_key: jax.Array,
      all_families: list[FamilyEntity],
  ) -> recoglab_base.ReCogLabQuestionAnswerBlock:
    """Generates a cross-family size comparison question."""

    _, subkey1, subkey2 = jax.random.split(prng_key, 3)

    # Initialize a QA block
    block = recoglab_base.ReCogLabQuestionAnswerBlock()

    # Choose family 1
    family1_index = int(jax.random.randint(subkey1, (), 0, len(all_families)))
    family1 = all_families[family1_index]

    # If cfg hop length is not set, use a random hop length between
    # 1 and half the number of families (to ensure distance)

    min_hop = max(1, len(all_families) // 2)
    if self.cfg.hop_length != -1:
      hop_length_to_use = self.cfg.hop_length
    else:
      hop_length_to_use = int(
          jax.random.randint(subkey2, (), min_hop, len(all_families))
      )

    family2_index = (all_families.index(family1) + hop_length_to_use) % len(
        all_families
    )
    family2 = all_families[family2_index]

    assert family1_index != family2_index

    # Add address to diambiguiate in case last name clashes
    block.prompt.append(
        f"Which family is larger, the {family1.text} family living on"
        f" {family1.address} or the {family2.text} family living on"
        f" {family2.address}? Answer with the family name of the larger family."
    )
    if len(family1.members) == len(family2.members):  # Tie
      block.answers = [family1.text, family2.text]  # Include both families
    else:
      larger_family = (
          family1.text
          if len(family1.members) > len(family2.members)
          else family2.text
      )
      block.answers = [larger_family]

    return block

  def _generate_family_member_age_comparison_question(
      self,
      prng_key: jax.Array,
      all_families: list[FamilyEntity],
  ) -> recoglab_base.ReCogLabQuestionAnswerBlock:
    """Generates a cross-family age comparison question."""

    _, subkey1, subkey2, subkey3, subkey4 = jax.random.split(prng_key, 5)

    # Initialize a QA block
    block = recoglab_base.ReCogLabQuestionAnswerBlock()

    # Choose family 1
    family1_index = int(jax.random.randint(subkey1, (), 0, len(all_families)))
    family1 = all_families[family1_index]

    # If cfg hop length is not set, use a random hop length between
    # 1 and half the number of families (to ensure distance)

    min_hop = max(1, len(all_families) // 2)
    if self.cfg.hop_length != -1:
      hop_length_to_use = self.cfg.hop_length
    else:
      hop_length_to_use = int(
          jax.random.randint(subkey2, (), min_hop, len(all_families))
      )

    family2_index = (all_families.index(family1) + hop_length_to_use) % len(
        all_families
    )
    family2 = all_families[family2_index]
    assert family1_index != family2_index

    member1 = family1.members[
        int(jax.random.randint(subkey3, (), 0, len(family1.members)))
    ]
    member2 = family2.members[
        int(jax.random.randint(subkey4, (), 0, len(family2.members)))
    ]

    if member1.age > member2.age:
      answer = member1.text
    elif member2.age > member1.age:
      answer = member2.text
    else:
      answer = member1.text if member1.text < member2.text else member2.text

    # Add address to diambiguiate in case last name clashes
    block.prompt.append(
        f"Who is older: {member1.text} from the {family1.text} family living on"
        f" {family1.address} or {member2.text} from the {family2.text} family"
        f" living on {family2.address}? If both are the same age, answer with"
        " the name that comes first alphabetically. Answer with the name."
    )
    block.answers = [answer]
    return block

  def _generate_family_member_hobby_comparison_question(
      self,
      prng_key: jax.Array,
      all_families: list[FamilyEntity],
  ) -> recoglab_base.ReCogLabQuestionAnswerBlock:
    """Generates a cross-family shared hobby question."""
    _, subkey1, subkey2, subkey3, subkey4 = jax.random.split(prng_key, 5)

    # Initialize a QA block
    block = recoglab_base.ReCogLabQuestionAnswerBlock()

    # Choose family 1
    family1_index = int(jax.random.randint(subkey1, (), 0, len(all_families)))
    family1 = all_families[family1_index]

    # If cfg hop length is not set, use a random hop length between
    # 1 and half the number of families (to ensure distance)

    min_hop = max(1, len(all_families) // 2)

    if self.cfg.hop_length != -1:
      hop_length_to_use = self.cfg.hop_length
    else:
      hop_length_to_use = int(
          jax.random.randint(subkey2, (), min_hop, len(all_families))
      )

    family2_index = (all_families.index(family1) + hop_length_to_use) % len(
        all_families
    )
    family2 = all_families[family2_index]

    assert family1_index != family2_index

    member1 = family1.members[
        int(jax.random.randint(subkey3, (), 0, len(family1.members)))
    ]
    member2 = family2.members[
        int(jax.random.randint(subkey4, (), 0, len(family2.members)))
    ]

    assert member1 != member2

    shared_hobbies = sorted(
        list(set(member1.hobbies).intersection(set(member2.hobbies)))
    )

    # Add address to diambiguiate in case last name clashes
    block.prompt.append(
        f"What hobbies do {member1.text} from the {family1.text} family living"
        f" on {family1.address} and {member2.text} from the"
        f" {family2.text} family living on {family2.address} share? List the"
        " hobbies in alphabetical order, separated by commas, or answer N/A if"
        " they share no hobbies."
    )
    block.answers = [", ".join(shared_hobbies) if shared_hobbies else "N/A"]
    return block


class FamilyBlock(recoglab_base.ReCogLabBlock):
  """Data class for a family-based information block."""

  def __init__(self, family: FamilyEntity):
    super().__init__()
    self.prompt.append(_pretty_print_family(family))


class FamilyModule(recoglab_base.ReCogLabModule):
  """Data class for family-based question answering."""


class FamilyModuleGenerator:
  """Generates a module of family-related questions."""

  def __init__(
      self,
      cfg: ml_collections.ConfigDict,
      jax_init_key: jax.Array,
      split: common_types.DatasetSplit,
  ):
    super().__init__()

    del jax_init_key
    self.cfg = cfg
    self.split = split

    self.family_data_generator = GenerateFamilyData(cfg, split)
    self.question_generator = FamilyQuestionGenerator(cfg)

  def generate_module(
      self,
      prng_key: jax.Array,
      ignore_list_entities: list[common_types.Entity] | None = None,
  ) -> FamilyModule:
    """Generates a complete FamilyModule.

    Args:
      prng_key: The random number generator key to use.
      ignore_list_entities: A list of entities to not use.

    Returns:
      A FamilyModule with the generated data.
    """

    module = FamilyModule()
    del ignore_list_entities  # Currently unused (families should be unique)

    prng_key, subkey = jax.random.split(prng_key)
    families = self.family_data_generator.generate_dataset(
        subkey, self.cfg.num_families
    )
    module.entities = families

    for family in families:
      module.blocks.append(FamilyBlock(family))

    module.answer_block = self.question_generator.generate(
        prng_key=prng_key,
        all_families=families,
    )
    return module
