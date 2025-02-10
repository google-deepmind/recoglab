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

"""Code for generating Filler for the ReCogLab dataset."""

import jax
import jax.numpy as jnp

from recoglab import common_types
from recoglab.modules import recoglab_base


# Data for filler

# Random text fillers
RANDOM_TEXT_FILLERS = [
    (
        'London was our present point of rest; we determined to remain several'
        ' months in this wonderful and celebrated city'
    ),
    (
        'We quitted London on the 27th of March, and remained a few days at'
        ' Windsor, rambling in its beautiful forest'
    ),
    (
        'We visited the tomb of the illustrious Hampden, and the field on which'
        ' that patriot fell'
    ),
    (
        'I enjoyed this scene; and yet my enjoyment was embittered both by the'
        ' memory of the past, and the anticipation of the future'
    ),
    (
        'I was formed for peaceful happiness'
        'During my youthful days discontent never visited my mind'
    ),
    'I am a blasted tree; the bolt has entered my soul',
    (
        'I felt then that I should survive to exhibit, what I shall soon cease'
        ' to be--a miserable spectacle of wrecked humanity, pitiable to others,'
        ' and intolerable to myself'
    ),
    (
        'We passed a considerable period at Oxford, rambling among its'
        ' environs, and endeavouring to identify every spot which might relate'
        ' to the most animating epoch of English history'
    ),
    (
        'The spirit of elder days found a dwelling here, and we delighted to'
        ' trace its footsteps'
    ),
    (
        'If these feelings had not found an imaginary gratification, the'
        ' appearance of the city had yet in itself sufficient beauty to obtain'
        ' our admiration'
    ),
    (
        'The colleges are ancient and picturesque; the streets are almost'
        ' magnificent; and the lovely Isis, which flows beside it through'
        ' meadows of exquisite verdure, is spread forth into a placid expanse'
        ' of waters, which reflects its majestic assemblage of towers, and'
        ' spires, and domes, embosomed among aged trees'
    ),
    (
        'We left Oxford with regret, and proceeded to Matlock, which was our'
        ' next place of rest'
    ),
    (
        'The country in the neighbourhood of this village'
        ' resembled, to a greater degree, the scenery of Switzerland; but'
        ' everything is on a lower scale, and the green hills want the crown of'
        ' distant white Alps, which always attend on the piny mountains of my'
        ' native country'
    ),
    (
        'We visited the wondrous cave, and the little cabinets of natural'
        ' history, where the curiosities are disposed in the same manner as in'
        ' the collections at Servox and Chamounix'
    ),
    (
        'The latter name made me tremble when pronounced by Henry; and I'
        ' hastened to quit Matlock, with which that terrible scene was thus'
        ' associated'
    ),
    (
        'From Derby, still journeying northward, we passed two months in'
        ' Cumberland and Westmoreland'
    ),
    'I could now almost fancy myself among the Swiss mountains',
    (
        'The little'
        ' patches of snow which yet lingered on the northern sides of the'
        ' mountains, the lakes, and the dashing of the rocky streams, were all'
        ' familiar and dear sights to me'
    ),
    (
        'Here also we made some acquaintances, who almost contrived to cheat me'
        ' into happiness'
    ),
    (
        'The delight of Clerval was proportionably greater than mine; his mind'
        ' expanded in the company of men of talent, and he found in his own'
        ' nature greater capacities and resources than he could have imagined'
        ' himself to have possessed while he associated with his inferiors'
    ),
    (
        '"I could pass my life here," said he to me; "and among these mountains'
        ' I should scarcely regret Switzerland and the Rhine"'
    ),
    (
        "But he found that a traveller's life is one that includes much pain"
        ' amidst its enjoyments'
    ),
    (
        'His feelings are for ever on the stretch; and when he begins to sink'
        ' into repose, he finds himself obliged to quit that on which he rests'
        ' in pleasure for something new, which again engages his attention, and'
        ' which also he forsakes for other novelties'
    ),
    (
        'We had scarcely visited the various lakes of Cumberland and'
        ' Westmoreland, and conceived an affection for some of the inhabitants,'
        ' when the period of our appointment with our Scotch friend approached,'
        ' and we left them to travel on'
    ),
    'For my own part I was not sorry',
    (
        'I had now neglected my promise for some time, and I feared the effects'
        " of the daemon's disappointment"
    ),
    'He might remain in Switzerland, and wreak his vengeance on my relatives',
    (
        'This idea pursued me, and tormented me at every moment from which I'
        ' might otherwise have snatched repose and peace'
    ),
    (
        'I waited for my letters with feverish impatience: if they were'
        ' delayed, I was miserable, and overcome by a thousand fears; and when'
        ' they arrived, and I saw the superscription of Elizabeth or my father,'
        ' I hardly dared to read and ascertain my fate'
    ),
    (
        'Sometimes I thought that the fiend followed me, and might expedite my'
        ' remissness by murdering my companion'
    ),
    (
        'When these thoughts possessed me, I would not quit Henry for a moment,'
        ' but followed him us his shadow, to protect him from the fancied rage'
        ' of his destroyer'
    ),
    (
        'I felt as if I had committed some great crime, the consciousness of'
        ' which haunted me'
    ),
    (
        'I was guiltless, but I had indeed drawn down a horrible curse upon my'
        ' head, as mortal as that of crime'
    ),
    (
        'I visited Edinburgh with languid eyes and mind; and yet that city'
        ' might have interested the most unfortunate being'
    ),
    (
        'Clerval did not like it so well as Oxford: for the antiquity of the'
        ' latter city was more pleasing to him'
    ),
    (
        'But the beauty and regularity of the new town of Edinburgh, its'
        ' romantic castle, and its environs, the most delightful in the world,'
        " Arthur's Seat, St. Bernard's Well, and the Pentland Hills,"
        ' compensated him for the change, and filled him with cheerfulness and'
        ' admiration'
    ),
    'But I was impatient to arrive at the termination of my journey',
    (
        "We left Edinburgh in a week, passing through Coupar, St. Andrew's, and"
        ' along the banks of the Tay, to Perth, where our friend expected us'
    ),
    (
        'But I was in no mood to laugh and talk with strangers, or enter into'
        ' their feelings or plans with the good humour expected from a guest;'
        ' and accordingly I told Clerval that I wished to make the tour of'
        ' Scotland alone'
    ),
    '"Do you," said I, "enjoy yourself, and let this be our rendezvous',
    (
        'I may be absent a month or two; but do not interfere with my motions,'
        ' I entreat you: leave me to peace and solitude for a short time; and'
        ' when I return, I hope it will be with a lighter heart, more congenial'
        ' to your own temper"'
    ),
    (
        'Henry wished to dissuade me; but, seeing me bent on this plan, ceased'
        ' to remonstrate'
    ),
    'He entreated me to write often',
    (
        '"I had rather be with you," he said, "in your solitary rambles, than'
        ' with these Scotch people, whom I do not know: hasten then, my dear'
        ' friend, to return, that I may again feel myself somewhat at home,'
        ' which I cannot do in your absence"'
    ),
    (
        'Having parted from my friend, I determined to visit some remote spot'
        ' of Scotland, and finish my work in solitude'
    ),
    (
        'I did not doubt but that the monster followed me, and would discover'
        ' himself to me when I should have finished, that he might receive his'
        ' companion'
    ),
    (
        'With this resolution I traversed the northern highlands, and fixed on'
        ' one of the remotest of the Orkneys as the scene of my labours'
    ),
    (
        'It was a place fitted for such a work, being hardly more than a rock,'
        ' whose high sides were continually beaten upon by the waves'
    ),
    (
        'The soil was barren, scarcely affording pasture for a few miserable'
        ' cows, and oatmeal for its inhabitants, which consisted of five'
        ' persons, whose gaunt and scraggy limbs gave tokens of their miserable'
        ' fare'
    ),
    (
        'Vegetables and bread, when they indulged in such luxuries, and even'
        ' fresh water, was to be procured from the main land, which was about'
        ' five miles distant'
    ),
    (
        'On the whole island there were but three miserable huts, and one of'
        ' these was vacant when I arrived'
    ),
    'This I hired',
    (
        'It contained but two rooms, and these exhibited all the squalidness of'
        ' the most miserable penury'
    ),
    (
        'The thatch had fallen in, the walls were unplastered, and the door was'
        ' off its hinges'
    ),
    (
        'I ordered it to be repaired, bought some furniture, and took'
        ' possession; an incident which would, doubtless, have occasioned some'
        ' surprise, had not all the senses of the cotters been benumbed by want'
        ' and squalid poverty'
    ),
    (
        'In this retreat I devoted the morning to labour; but in the evening,'
        ' when the weather permitted, I walked on the stony beach of the sea,'
        ' to listen to the waves as they roared and dashed at my feet'
    ),
    'It was a monotonous yet ever-changing scene',
    (
        'I thought of Switzerland; it was far different from this desolate and'
        ' appalling landscape'
    ),
    (
        'Its hills are covered with vines, and its cottages are scattered'
        ' thickly in the plains'
    ),
    (
        'Its fair lakes reflect a blue and gentle sky; and, when troubled by'
        ' the winds, their tumult is but as the play of a lively infant, when'
        ' compared to the roarings of the giant ocean'
    ),
    (
        'In this manner I distributed my occupations when I first arrived; but,'
        ' as I proceeded in my labour, it became every day more horrible and'
        ' irksome to me'
    ),
    (
        'Sometimes I could not prevail on myself to enter my laboratory for'
        ' several days; and at other times I toiled day and night in order to'
        ' complete my work'
    ),
    'It was, indeed, a filthy process in which I was engaged',
    (
        'During my first experiment, a kind of enthusiastic frenzy had blinded'
        ' me to the horror of my employment; my mind was intently fixed on the'
        ' consummation of my labour, and my eyes were shut to the horror of my'
        ' proceedings'
    ),
    (
        'But now I went to it in cold blood, and my heart often sickened at the'
        ' work of my hands'
    ),
    (
        'Thus situated, employed in the most detestable occupation, immersed in'
        ' a solitude where nothing could for an instant call my attention from'
        ' the actual scene in which I was engaged, my spirits became unequal; I'
        ' grew restless and nervous'
    ),
    'Every moment I feared to meet my persecutor',
    (
        'Sometimes I sat with my eyes fixed on the ground, fearing to raise'
        ' them, lest they should encounter the object which I so much dreaded'
        ' to behold'
    ),
    (
        'I feared to wander from the sight of my fellow-creatures, lest when'
        ' alone he should come to claim his companion'
    ),
    (
        'In the meantime I worked on, and my labour was already considerably'
        ' advanced'
    ),
    (
        'I looked towards its completion with a tremulous and eager hope, which'
        ' I dared not trust myself to question, but which was intermixed with'
        ' obscure forebodings of evil, that made my heart sicken in my bosom'
    ),
    (
        'It is a truth universally acknowledged, that a single man in'
        ' possession of a good fortune must be in want of a wife'
    ),
    (
        'However little known the feelings or views of such a man may be on his'
        ' first entering a neighbourhood, this truth is so well fixed in the'
        ' minds of the surrounding families, that he is considered as the'
        ' rightful property of some one or other of their daughters'
    ),
    '"Is he married or single?"',
    (
        'Oh, single, my dear, to be sure! A single man of large fortune; four'
        ' or five thousand a year'
    ),
    'What a fine thing for our girls!',
    'How so? how can it affect them?',
    '"My dear Mr. Bennet," replied his wife, "how can you be so tiresome?',
    'You must know that I am thinking of his marrying one of them',
    'Is that his design in settling here?',
    (
        '"Design? Nonsense, how can you talk so! But it is very likely that he'
        ' may fall in love with one of them, and therefore you must visit him'
        ' as soon as he comes"'
    ),
    'I see no occasion for that',
    (
        'You and the girls may go—or you may send'
        ' them by themselves, which perhaps will be still better; for as you'
        ' are as handsome as any of them, Mr. Bingley might like you the best'
        ' of the party'
    ),
    'My dear, you flatter me',
    (
        'I certainly have had my share of beauty, but I do not pretend to be'
        ' anything extraordinary now'
    ),
    (
        'When a woman has five grown-up daughters, she ought to give over'
        ' thinking of her own beauty'
    ),
    'In such cases, a woman has not often much beauty to think of',
    (
        '"But, my dear, you must indeed go and see Mr. Bingley when he comes'
        ' into the neighbourhood"'
    ),
    'It is more than I engage for, I assure you"',
    'But consider your daughters',
    'Only think what an establishment it would be for one of them',
    (
        'Sir William and Lady Lucas are determined to go, merely on that'
        ' account; for in general, you know, they visit no new-comers'
    ),
    (
        'Indeed you must go, for it will be impossible for us to visit him, if'
        ' you do not'
    ),
    'You are over scrupulous, surely',
    (
        'I dare say Mr. Bingley will be very'
        ' glad to see you; and I will send a few lines by you to assure him of'
        ' my hearty consent to his marrying whichever he chooses of the'
        ' girls—though I must throw in a good word for my little Lizzy'
    ),
    'I desire you will do no such thing',
    (
        'Lizzy is not a bit better than the others: and I am sure she is not'
        ' half so handsome as Jane, nor half so good-humoured as Lydia'
    ),
    'But you are always giving her the preference',
    (
        '"They have none of them much to recommend them," replied he: "they are'
        ' all silly and ignorant like other girls; but Lizzy has something more'
        ' of quickness than her sisters"'
    ),
    'Mr. Bennet, how can you abuse your own children in such a way?',
    'You take delight in vexing me',
    'You have no compassion on my poor nerves',
    'You mistake me, my dear',
    'I have a high respect for your nerves',
    'They are my old friends',
    (
        'I have heard you mention them with consideration these twenty years at'
        ' least'
    ),
    'Ah, you do not know what I suffer',
    (
        '"But I hope you will get over it, and live to see many young men of'
        ' four thousand a year come into the neighbourhood"'
    ),
    (
        '"It will be no use to us, if twenty such should come, since you will'
        ' not visit them"'
    ),
    (
        '"Depend upon it, my dear, that when there are twenty, I will visit'
        ' them all"'
    ),
    (
        'Mr. Bennet was so odd a mixture of quick parts, sarcastic humour,'
        ' reserve, and caprice, that the experience of three-and-twenty years'
        ' had been insufficient to make his wife understand his character'
    ),
    'Her mind was less difficult to develope',
    (
        'She was a woman of mean understanding, little information, and'
        ' uncertain temper'
    ),
    'When she was discontented, she fancied herself nervous',
    (
        'The business of her life was to get her daughters married: its solace'
        ' was visiting and news'
    ),
    (
        'Not all that Mrs. Bennet, however, with the assistance of her five'
        ' daughters, could ask on the subject, was sufficient to draw from her'
        ' husband any satisfactory description of Mr. Bingley'
    ),
    (
        'They attacked him in various ways, with barefaced questions, ingenious'
        ' suppositions, and distant surmises; but he eluded the skill of them'
        ' all; and they were at last obliged to accept the second-hand'
        ' intelligence of their neighbour, Lady Lucas'
    ),
    'Her report was highly favourable',
    'Sir William had been delighted with him',
    (
        'He was quite young, wonderfully handsome, extremely agreeable, and, to'
        ' crown the whole, he meant to be at the next assembly with a large'
        ' party'
    ),
    'Nothing could be more delightful!',
    (
        'To be fond of dancing was a certain step towards falling in love; and'
        ' very lively hopes of Mr. Bingley’s heart were entertained'
    ),
    (
        '"If I can but see one of my daughters happily settled at Netherfield,"'
        ' said Mrs. Bennet to her husband, "and all the others equally well'
        ' married, I shall have nothing to wish for"'
    ),
    (
        'In a few days Mr. Bingley returned Mr. Bennet’s visit, and sat about'
        ' ten minutes with him in his library'
    ),
    (
        'He had entertained hopes of being admitted to a sight of the young'
        ' ladies, of whose beauty he had heard much; but he saw only the'
        ' father'
    ),
    (
        'The ladies were somewhat more fortunate, for they had the advantage of'
        ' ascertaining, from an upper window, that he wore a blue coat and rode'
        ' a black horse'
    ),
    (
        'An invitation to dinner was soon afterwards despatched; and already'
        ' had Mrs. Bennet planned the courses that were to do credit to her'
        ' housekeeping, when an answer arrived which deferred it all'
    ),
    (
        'Mr. Bingley was obliged to be in town the following day, and'
        ' consequently unable to accept the honour of their invitation, etc'
    ),
    'Mrs. Bennet was quite disconcerted',
    (
        'She could not imagine what business he could have in town so soon'
        ' after his arrival in Hertfordshire; and she began to fear that he'
        ' might always be flying about from one place to another, and never'
        ' settled at Netherfield as he ought to be'
    ),
]

# Templates for entity fillers
PERSON_ENTITY_FILLERS = [
    '{ENTITY} is very smart',
    '{ENTITY} has good teeth',
    '{ENTITY} often likes to sit in the shade and read a book',
    '{ENTITY} cannot imagine not living in the country',
    '{ENTITY} is very good at playing chess',
    '{ENTITY} enjoys movies',
    '{ENTITY} can play the piano',
    'In the eveinings {ENTITY} likes to sit by the fire and read a book',
    'After work {ENTITY} grabs a snack and then goes for a walk',
    'You can find {ENTITY} at the coffee shop every morning',
    '{ENTITY} was at the park yesterday',
    'Whether you see {ENTITY} or not, they are always there',
    'Yesterday, {ENTITY} went snowboarding',
    '{ENTITY} is a competitive person who does not like to lose',
    'The only person who can do that is {ENTITY}',
    'Saving money is {ENTITY}’s favorite hobby',
    '{ENTITY} is a good person',
    'Piano is something that {ENTITY} is very good at',
    '{ENTITY} is a very good cook',
    'Forever and always, {ENTITY} is a good person',
    'The only person who can eat that much is {ENTITY}',
    'Eating alone is something that {ENTITY} does not like to do',
    (
        'Creating art for the sake of art is something that {ENTITY} does not'
        ' like to do'
    ),
    'Passing time with friends is something that {ENTITY} likes to do',
]
OBJECT_ENTITY_FILLERS = [
    '{ENTITY} is expensive',
    '{ENTITY} is a good thing to have around',
    'I saw a {ENTITY} yesterday',
    'I like to look at {ENTITY} when I am sad',
    'Wherever you go, you can find {ENTITY}',
    'Seeing {ENTITY} makes me happy',
    'You can find {ENTITY} in the park',
    'Eating {ETITY} is a bad idea',
    'Wherever you go, you can find {ENTITY}',
    'Placing {ENTITY} in the sun is a good idea',
    'You can put {ENTITY} on anything',
    'Having {ENTITY} is a bad idea',
    '{ENTITY} is blue',
    '{ENTITY} is green',
    '{ENTITY} is red',
    '{ENTITY} is yellow',
    '{ENTITY} is purple',
    '{ENTITY} is orange',
    '{ENTITY} is brown',
    '{ENTITY} is black',
    '{ENTITY} is white',
]


def generate_random_text(
    num_lines: int, subkey: jax.Array
) -> list[recoglab_base.ReCogLabBlock]:
  """Generates random text."""
  # Randomly sample from RANDOM_TEXT_FILLERS
  filler_blocks = []
  for _ in range(num_lines):
    block = recoglab_base.ReCogLabBlock()
    _, subkey = jax.random.split(subkey)
    i = jax.random.choice(
        subkey,
        jnp.arange(len(RANDOM_TEXT_FILLERS)),
        replace=False,
        shape=(),
    )
    filler_line = RANDOM_TEXT_FILLERS[i]
    block.prompt = filler_line
    filler_blocks.append(block)
  return filler_blocks


def generate_entity_filler(
    num_lines: int,
    entities: list[common_types.Entity],
    entity_type: str,
    subkey: jax.Array,
) -> list[recoglab_base.ReCogLabBlock]:
  """Generates entity filler."""
  if entity_type in ['people', 'baby-names',]:
    templates = PERSON_ENTITY_FILLERS
  elif entity_type in ['basic_objects', 'plural_nouns']:
    templates = OBJECT_ENTITY_FILLERS
  else:
    raise NotImplementedError(
        f'Need to implement this entity type: {entity_type}'
    )
  filler_blocks = []
  for _ in range(num_lines):
    block = recoglab_base.ReCogLabBlock()
    _, subkey = jax.random.split(subkey)
    ent_i = jax.random.choice(
        subkey,
        jnp.arange(len(entities)),
        replace=False,
        shape=(),
    )
    entity = entities[ent_i]
    _, subkey = jax.random.split(subkey)
    template_i = jax.random.choice(
        subkey,
        jnp.arange(len(templates)),
        replace=False,
        shape=(),
    )
    template = templates[template_i]
    filler_line = template.format(ENTITY=entity.text)
    block.prompt = filler_line
    filler_blocks.append(block)
  return filler_blocks
