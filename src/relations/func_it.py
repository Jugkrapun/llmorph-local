from llm_runner import run_template_gpt
from file_handler import load_json
from .func_base import FuncIT
import random
import string
import math
from nltk.tokenize import sent_tokenize, word_tokenize
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import re
# Custom negation feature
import spacy
import os
from openai import OpenAI
from typing import Set, Dict, List

nlp = spacy.load("en_core_web_trf")

RANDOM_SENTENCES = load_json("./resources/random_sentences.json")
RANDOM_WORDS = load_json("./resources/random_words.json")

# MR list
# 1	Replace characters with random
# 2	Delete characters
# 3	L33t format
# 4	Add characters
# 5	Add spaces
# 6	Swap characters
# 7	Shuffle characters
# 8	Synonym replacement
# 9	Word insertion
# 10	Antonym replacement (output: difference)
# 19	Randomise sentence order
# 25	Replace keywords with random (output: difference)
# 34	Remove keywords (output: difference)
# 49	No change
# 51	Paraphrasing
# 57	Declarative to interrogative sentence
# 77	Generate premise from hypothesis and add to premise
# 78	Generate hypothesis from premise and add to hypothesis
# 79	Generate hypothesis from hypothesis and use as hypothesis (output: not opposite)
# 80	Generate hypothesis from premise and use as premise (output: not opposite)
# 84	Add random sentence
# 102	Capitalisation of all
# 120	Back translate
# 126	Keyboard errors
# 127	Misspell
# 128	OCR errors
# 136	Passive/active voice
# 137	Replace word with another in same category
# 141	Swap symmetric entities
# 142	Swap asymmetric entities (output: opposite relation)
# 149	Singular/plural
# 150	Replace . with !
# 151	Add emphasis words
# 152	Negate (output: difference)
# 154	Capitalise important words
# 155	Tense change

class CleanText():
    def clean_text(self, text):
        lowercased = text.lower()
        no_punctuation = ''.join(char for char in lowercased if char not in string.punctuation)
        cleaned = ' '.join(no_punctuation.split())
        return cleaned

class ITBase(FuncIT):
    def __init__(self, transform_indices=[[0]], multi_input=False):
        self.transform_indices = transform_indices
        self.multi_input = multi_input

    def transform_targeted(self, input: list, n: int, transformation) -> str:
        transform_target = input if self.multi_input else input[n]
        return transformation(transform_target)

    def transform_input(self, input: list, transformation):
        unique_indices = set(index for sublist in self.transform_indices for index in sublist)

        # get transformed values for each of any specified index
        transformed_values = [None] * len(input)
        for i in unique_indices:
            transformed_values[i] = self.transform_targeted(input, i, transformation)
        
        # generate the follow-up inputs from those transformed values
        output_values = []
        for indices in self.transform_indices:
            output_value = input.copy()
            for i in indices:
                output_value[i] = transformed_values[i]
            output_values.append(output_value)

        return output_values

class SingleInputTransformer(ITBase):
    def __init__(self, transform_indices=[[0]]):
        super().__init__(transform_indices, False)

# MR-49
class ITNone(FuncIT):
    def __init__(self, *args, **kwargs):
        pass

    def input_transformation(self, input: list):
        return [input]

class GPTRunner():
    def run_gpt(self, input, prompt_template, examples=[]):
        if not isinstance(input, list):
            input = [input]
        return run_template_gpt(input, prompt_template, examples)

class ITGPT(ITBase):
    def __init__(self, prompt_template: str, examples=[], transform_indices=[[0]], multi_input=False):
        super().__init__(transform_indices, multi_input)
        self.prompt_template = prompt_template
        self.examples = examples

    def run_gpt(self, input):
        if not isinstance(input, list):
            input = [input]
        return run_template_gpt(input, self.prompt_template, self.examples)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.run_gpt)

class ITGPTSentence(ITGPT):
    def sentence_transform(self, input: list):
        input_sentences = sent_tokenize(input)
        output = [self.run_gpt(sentence) for sentence in input_sentences]
        return ' '.join(output)
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.sentence_transform)

class ITGPTConcatInf(ITGPT):
    def input_transformation(self, input: list):
        output = input.copy()
        transform_target = output if self.multi_input else output[self.transform_indices]
        output[self.transform_indices[0][0]] = output[self.transform_indices[0][0]] + " " + self.run_gpt(transform_target) # hardcoded...
        return [output]

class ITConcat(SingleInputTransformer):
    def __init__(self, addition: str, transform_indices=[[0]]):
        super().__init__(transform_indices)
        self.addition = addition

    def concat(self, input_val):
        return input_val + " " + self.addition

    def input_transformation(self, input: list):
        return self.transform_input(input, self.concat)
# MR-84    
class ITConcatRandomSentence_edit(SingleInputTransformer):
    def __init__(self, transform_indices=[[0]], rand_seed=42):
        super().__init__(transform_indices)
        self.data = RANDOM_SENTENCES
        self.rand = random.Random(rand_seed)
    
    def concat_random(self, input_val: str) -> str:
        cleaned_input = input_val.rstrip()
        if not cleaned_input:
            return input_val
    # Sample a random sentence from the original data source and remove white space
        random_datum = self.rand.choice(self.data).strip()
        
        # 1. Prevent prompt hijacking: Check for terminal prompt markers (e.g., "\nAnswer:")
        prompt_markers = ["\nAnswer:", "\nChoices:", "\nQuestion:"]
        for marker in prompt_markers:
            if marker in cleaned_input:
                prefix, suffix = cleaned_input.rsplit(marker, 1)
                # Ensure the preceding text ends with valid punctuation
                glue = "" if prefix.rstrip().endswith((".", "!", "?")) else "."
                # Insert the random sentence before the prompt marker
                return f"{prefix.rstrip()}{glue} {random_datum}\n{marker.strip()}{suffix}"

        # 2. Prevent run-on sentences: Add a period if the original input lacks closing punctuation
        has_punctuation = cleaned_input[-1] in {".", "!", "?"}
        punctuation_glue = "" if has_punctuation else "."

        # Append strictly to the end of the input according to the original definition
        return f"{cleaned_input}{punctuation_glue} {random_datum}"

    def input_transformation(self, input: list):
        return self.transform_input(input, self.concat_random)

# original MR-84
class ITConcatRandomSentence(SingleInputTransformer):
    def __init__(self, transform_indices=[[0]], rand_seed=42):
        super().__init__(transform_indices)
        self.data = RANDOM_SENTENCES
        self.rand = random.Random(rand_seed)

    def concat_random(self, input_val):
        random_datum = self.rand.choice(self.data)
        return input_val + " " + random_datum
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.concat_random)


# not used
class ITSentiment(CleanText, SingleInputTransformer):
    def get_sentiment(self, inputs): # temporary
        prompt_template = "You are a sentiment analysis tool. Given a sentence, say if it is 'positive' or 'negative' or 'neutral', nothing else.\nWhat is the sentiment of the following sentence?\n\"{INPUT_0}\"\nOnly write a one-word answer."
        response = run_template_gpt([inputs], prompt_template)
        return self.clean_text(response)
 
# not used
class ITGroupBySentiment(ITSentiment):
    def group_sentiments(self, input_val):
        sentences = [sentence.strip() + '.' for sentence in input_val.split('.') if sentence]
        if sentences and not input_val.endswith('.'):
            sentences[-1] = sentences[-1].rstrip('.')
        
        sentiments = {sentence: self.get_sentiment(sentence) for sentence in sentences}
        grouped = {'positive': [], 'negative': [], 'neutral': []}
        for sentence, sentiment in sentiments.items():
            grouped[sentiment].append(sentence)
        return ' '.join([' '.join(grouped[sentiment]) for sentiment in ['positive', 'negative', 'neutral'] if grouped[sentiment]])

    def input_transformation(self, input: list):
        return self.transform_input(input, self.group_sentiments)

# not used
class ITGPTBackTranslate(SingleInputTransformer):
    def back_translate(self, input_val):
        prompt_template_to = "Translate the following into Korean:\n\"{INPUT_0}\"\nOnly output the tranlated text."
        response_to = run_template_gpt([input_val], prompt_template_to)
        prompt_template_from = "Translate the following into English:\n\"{INPUT_0}\"\nOnly output the tranlated text."
        response_from = run_template_gpt([response_to], prompt_template_from)
        return response_from

    def input_transformation(self, input: list):
        return self.transform_input(input, self.back_translate)

class ITPermuteInputs(FuncIT):
    # a permutation list of which element to map to where, e.g. [2,0,1]
    def __init__(self, permute_to: list):
        self.permute_to = permute_to

    def input_transformation(self, input: list):
        output = [''] * len(input)
        for val, n in zip(input, self.permute_to):
            output[n] = val
        return [output]

# MR-102
class ITCapitalisation(SingleInputTransformer):
    def capitalise(self, input_val):
        return input_val.upper()

    def input_transformation(self, input: list):
        return self.transform_input(input, self.capitalise)

# MR-150
class ITReplacePeriodWithExclamation(SingleInputTransformer):
    def replace_period_with_exclamation(self, input_val):
        new_val = input_val.replace('.', '!')
        if not new_val.endswith('!'):
            new_val += '!'
        return new_val

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_period_with_exclamation)


class SingleInputRandomBase(SingleInputTransformer):
    def __init__(self, transform_indices=[[0]], rand_seed=42, replace_perc=0.1, **kwargs):
        super().__init__(transform_indices)
        self.rand = random.Random(rand_seed)
        self.replace_perc = replace_perc

# MR-19
class ITRandomiseSentenceOrder(SingleInputRandomBase):
    def randomise_sentences(self, input_val):
        sentences = sent_tokenize(input_val)
        self.rand.shuffle(sentences)
        return ' '.join(sentences)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.randomise_sentences)

# not used
class ITRandomiseWordOrder(SingleInputRandomBase):
    def randomise_words(self, input_val):
        words = word_tokenize(input_val)
        words_without_punct = [word for word in words if word not in string.punctuation]
        self.rand.shuffle(words_without_punct)
        return ' '.join(
            word if word in string.punctuation else words_without_punct.pop(0)
            for word in words
        )

    def input_transformation(self, input: list):
        return self.transform_input(input, self.randomise_words)

# not used
class ITRandomiseWordOrderInSentence(SingleInputRandomBase):
    def shuffle_sentence(self, sentence):
        words = word_tokenize(sentence)
        words_without_punct = [word for word in words if word not in string.punctuation]
        self.rand.shuffle(words_without_punct)
        return ' '.join(
            word if word in string.punctuation else words_without_punct.pop(0)
            for word in words
        )

    def shuffle_text(self, text):
        sentences = sent_tokenize(text)
        return ' '.join(self.shuffle_sentence(sentence) for sentence in sentences)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.shuffle_text)


# base class for character, word and sentence transformations, WE CAN CONFIG THIS!
class ObjectRandomBase(SingleInputRandomBase):
    def __init__(self, transform_indices=[[0]], replace_perc=0.1, rand_seed=42):
        super().__init__(
            transform_indices=transform_indices,
            rand_seed=rand_seed,
            replace_perc=replace_perc, # Change original owner of replace_perc to SingleInputRandomBase
        )
    
    def transform_function(self, text):
        tokens = self.tokenise(text)
        num_mutated = math.ceil((len(tokens) - 1) * self.replace_perc) # don't do all
        ids = self.rand.sample(range(len(tokens)), num_mutated)
        new_text = self.object_transform(ids, tokens)
        return self.join_tokens(new_text)
    
    def tokenise(self, text):
        pass

    def join_tokens(self, tokens):
        pass
    
    def object_transform(self, ids: list, text):
        pass

    def input_transformation(self, input: list):
        return self.transform_input(input, self.transform_function)


class CharacterRandomBase(ObjectRandomBase):
    def tokenise(self, text):
        return list(text)
    
    def join_tokens(self, tokens):
        return ''.join(tokens)
    
    def transform_function(self, text):
        # only modifies tokens that are letters
        tokens = self.tokenise(text)
        letter_ids = [i for i, token in enumerate(tokens) if token.isalpha()]
        num_mutated = math.ceil(len(letter_ids) * self.replace_perc)
        ids = self.rand.sample(letter_ids, num_mutated)
        new_text = self.object_transform(ids, tokens)

        return self.join_tokens(new_text)

# MR-1
class ITReplaceCharacters_edit(CharacterRandomBase):
    # Logical operators, quantifiers, and negations that alter core semantics if mutated
    PROTECTED_WORDS: Set[str] = {
        "not", "no", "never", "none", "neither", "nor", 
        "hardly", "scarcely", "barely", "all", "every"
    }

    def _get_protected_indices(self, raw_text: str) -> Set[int]:
        doc = nlp(raw_text)
        protected_indices: Set[int] = set()

        # 1. Protect all named entities
        for ent in doc.ents:
            protected_indices.update(range(ent.start_char, ent.end_char))

        # 2. Protect Proper Nouns, Negations, Numbers, Symbols, Punctuation, Whitespaces, and Acronyms
        for token in doc:
            is_proper_noun = token.pos_ == "PROPN" or token.tag_ in {"NNP", "NNPS"}
            is_negation = token.lower_ in self.PROTECTED_WORDS or token.dep_ == "neg"
            is_numeric = token.like_num or token.pos_ == "NUM"
            is_symbol_or_punct = token.is_punct or token.pos_ in {"SYM", "X"}
            is_acronym = token.text.isupper() and len(token.text) > 1
            is_whitespace = token.is_space

            if (is_proper_noun or is_negation or is_numeric or 
                is_symbol_or_punct or is_acronym or is_whitespace):
                protected_indices.update(range(token.idx, token.idx + len(token.text)))

        # 3. Protect full-term definitions preceding parenthesized acronyms
        # Example pattern: "Multiple-choice question answering (MCQA)"
        for i in range(len(doc) - 3):
            if doc[i + 1].text == "(" and doc[i + 2].text.isupper() and doc[i + 3].text == ")":
                curr = i
                while curr >= 0 and (doc[curr].pos_ in {"NOUN", "PROPN", "ADJ"} or doc[curr].text == "-"):
                    protected_indices.update(range(doc[curr].idx, doc[curr].idx + len(doc[curr].text)))
                    curr -= 1

        return protected_indices

    def object_transform(self, ids: List[int], text: List[str]) -> List[str]:
        raw_text = "".join(text)
        protected_indices = self._get_protected_indices(raw_text)

        valid_ids = [i for i in ids if i not in protected_indices]

        for i in valid_ids:
            original_char = text[i]

            # Skip non-alphabetic characters
            if not original_char.isalpha():
                continue

            # Generate a different letter, preserving original casing
            new_char = chr(self.rand.randint(97, 122))
            while new_char == original_char.lower():
                new_char = chr(self.rand.randint(97, 122))

            text[i] = new_char.upper() if original_char.isupper() else new_char

        return text

# original MR-1
class ITReplaceCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = chr(self.rand.randint(97, 122))
        return text

# MR-2
class ITDeleteCharacters_edit(CharacterRandomBase):
    # Logical operators, quantifiers, and negations that alter core semantics if mutated
    PROTECTED_WORDS: Set[str] = {
        "not", "no", "never", "none", "neither", "nor", 
        "hardly", "scarcely", "barely", "all", "every"
    }
    # Minimum character length a token must have before allowing character deletion
    MIN_TOKEN_LENGTH: int = 5

    def _get_protected_indices(self, raw_text: str) -> Set[int]:
        doc = nlp(raw_text)
        protected_indices: Set[int] = set()

        # 1. Protect all named entities
        for ent in doc.ents:
            protected_indices.update(range(ent.start_char, ent.end_char))
        for token in doc:
            is_proper_noun = token.pos_ == "PROPN" or token.tag_ in {"NNP", "NNPS"}
            is_negation = token.lower_ in self.PROTECTED_WORDS or token.dep_ == "neg"
            is_numeric = token.like_num or token.pos_ == "NUM"
            is_symbol_or_punct = token.is_punct or token.pos_ in {"SYM", "X"}
            is_acronym = token.text.isupper() and len(token.text) > 1
            is_whitespace = token.is_space
            is_too_short = len(token.text) < self.MIN_TOKEN_LENGTH

            if (is_proper_noun or is_negation or is_numeric or is_symbol_or_punct or 
                is_acronym or is_whitespace or is_too_short):
                protected_indices.update(range(token.idx, token.idx + len(token.text)))

        # 3. Protect full-term definitions preceding parenthesized acronyms
        # Example pattern: "Multiple-choice question answering (MCQA)"
        for i in range(len(doc) - 3):
            if doc[i + 1].text == "(" and doc[i + 2].text.isupper() and doc[i + 3].text == ")":
                curr = i
                while curr >= 0 and (doc[curr].pos_ in {"NOUN", "PROPN", "ADJ"} or doc[curr].text == "-"):
                    protected_indices.update(range(doc[curr].idx, doc[curr].idx + len(doc[curr].text)))
                    curr -= 1

        return protected_indices

    def object_transform(self, ids: List[int], text: List[str]) -> List[str]:
        raw_text = "".join(text)
        protected_indices = self._get_protected_indices(raw_text)

        # Filter indices generated by CharacterRandomBase against the protected set
        valid_ids = [i for i in ids if i not in protected_indices]

        for i in valid_ids:
            # Explicit guard: never delete whitespace or control characters
            if text[i].isspace():
                continue

            # Delete the character
            text[i] = ""

        return text

# original MR-2
class ITDeleteCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = ''
        return text

# MR-4
class ITAddCharacters_edit(CharacterRandomBase):
    def _get_protected_indices(self, raw_text: str) -> Set[int]:
        doc = nlp(raw_text)
        protected_indices: Set[int] = set()

        # 1. Protect all named entities
        for ent in doc.ents:
            protected_indices.update(range(ent.start_char, ent.end_char))

        # 2. Protect numbers, symbols, punctuation, and uppercase acronyms
        for token in doc:
            if token.pos_ in {"NUM", "SYM", "X"} or token.is_punct:
                protected_indices.update(range(token.idx, token.idx + len(token.text)))
            elif token.text.isupper() and len(token.text) > 1:
                protected_indices.update(range(token.idx, token.idx + len(token.text)))

        # 3. Protect full-term definitions preceding parenthesized acronyms (e.g., "Multiple-choice question answering (MCQA)")
        for i in range(len(doc) - 3):
            if doc[i + 1].text == "(" and doc[i + 2].text.isupper() and doc[i + 3].text == ")":
                curr = i
                while curr >= 0 and (doc[curr].pos_ in {"NOUN", "PROPN", "ADJ"} or doc[curr].text == "-"):
                    protected_indices.update(range(doc[curr].idx, doc[curr].idx + len(doc[curr].text)))
                    curr -= 1

        return protected_indices

    def object_transform(self, ids: List[int], text: List[str]) -> List[str]:
        raw_text = "".join(text)
        protected_indices = self._get_protected_indices(raw_text)

        for i in ids:
            # Shield protected indices and never append a character to whitespace
            if i not in protected_indices and not text[i].isspace():
                text[i] = text[i] + chr(self.rand.randint(97, 122))

        return text

# original MR-4
class ITAddCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = text[i] + chr(self.rand.randint(97, 122))
        return text

# MR-3
class ITLeetFormat_edit(CharacterRandomBase):
    # Leet mapping table
    LEET_DICT: Dict[str, str] = {
        'a': '4', 'A': '4',
        'e': '3', 'E': '3',
        'i': '1', 'I': '1',
        'o': '0', 'O': '0',
        't': '7', 'T': '7'
    }

    def _get_protected_indices(self, raw_text: str) -> Set[int]:
        doc = nlp(raw_text)
        protected_indices: Set[int] = set()
       
        # Prevent digit collision across spaces (e.g., prevents "room 10" -> "r00m 10")
        for i, char in enumerate(raw_text):
            if char.isdigit():
                # Look backwards past whitespace
                k = i - 1
                while k >= 0 and raw_text[k].isspace():
                    k -= 1
                if k >= 0:
                    protected_indices.add(k)

                # Look forwards past whitespace
                k = i + 1
                while k < len(raw_text) and raw_text[k].isspace():
                    k += 1
                if k < len(raw_text):
                    protected_indices.add(k)

        return protected_indices

    def object_transform(self, ids: List[int], text: List[str]) -> List[str]:
        raw_text = "".join(text)
        protected_indices = self._get_protected_indices(raw_text)

        for i in ids:
            if i not in protected_indices:
                text[i] = self.LEET_DICT.get(text[i], text[i])

        return text

# original MR-3
class ITLeetFormat(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        leet_dict = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 't': '7'}
        for i in ids:
            text[i] = leet_dict.get(text[i], text[i])
        return text

# MR-5
class ITAddSpaces(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = text[i] + ' '
        return text

# MR-6
class ITSwapCharacters_edit(CharacterRandomBase):
    def _get_protected_indices(self, raw_text: str) -> Set[int]:
        doc = nlp(raw_text)
        protected_indices: Set[int] = set()

        # 1. Named Entities (PERSON, ORG, GPE, DATE, etc.)
        for ent in doc.ents:
            protected_indices.update(range(ent.start_char, ent.end_char))

        # 2. Numbers, symbols, punctuation, and uppercase acronyms
        for token in doc:
            if token.pos_ in {"NUM", "SYM", "X"} or token.is_punct:
                protected_indices.update(range(token.idx, token.idx + len(token.text)))
            elif token.text.isupper() and len(token.text) > 1:
                protected_indices.update(range(token.idx, token.idx + len(token.text)))

        # 3. Full-term definitions preceding parenthesized acronyms (e.g., "Multiple-choice question answering (MCQA)")
        for i in range(len(doc) - 3):
            if doc[i + 1].text == "(" and doc[i + 2].text.isupper() and doc[i + 3].text == ")":
                curr = i
                while curr >= 0 and (doc[curr].pos_ in {"NOUN", "PROPN", "ADJ"} or doc[curr].text == "-"):
                    protected_indices.update(range(doc[curr].idx, doc[curr].idx + len(doc[curr].text)))
                    curr -= 1

        return protected_indices

    def object_transform(self, ids: List[int], text: List[str]) -> List[str]:
        raw_text = "".join(text)
        protected_indices = self._get_protected_indices(raw_text)

        last_swapped_idx = -2

        # Sort indices to process sequentially and avoid cascading reversals
        for i in sorted(ids):
            # Guard against swapping adjacent indices that were just modified
            if i <= last_swapped_idx + 1:
                continue

            if i < len(text) - 1:
                # Both positions must be unprotected and non-whitespace
                both_unprotected = (i not in protected_indices) and ((i + 1) not in protected_indices)
                both_non_space = (not text[i].isspace()) and (not text[i + 1].isspace())

                if both_unprotected and both_non_space:
                    text[i], text[i + 1] = text[i + 1], text[i]
                    last_swapped_idx = i

        return text

#  original MR-6
class ITSwapCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            if i < len(text) - 1:
                text[i], text[i + 1] = text[i + 1], text[i]
        return text

class WordRandomBase(ObjectRandomBase):
    def tokenise(self, text):
        return word_tokenize(text)
    
    def join_tokens(self, tokens):
        return ' '.join(tokens)

class ITRandomiseCharacterOrderInWord(WordRandomBase):
    def object_transform(self, ids: list, text):
        for i in ids:
            text[i] = ''.join(self.rand.sample(text[i], len(text[i])))
        return text

# MR-7
class ITRandomiseCharacterOrderInWordKeepingEnds_edit(WordRandomBase):
    # Lexical negation and quantification terms critical to premise truth values
    PROTECTED_WORDS: Set[str] = {
        "not", "no", "never", "none", "neither", "nor",
        "hardly", "scarcely", "barely", "all", "every"
    }

    def _get_protected_token_indices(self, doc) -> Set[int]:
        protected_ids: Set[int] = set()

        # 1. Named Entities (PERSON, ORG, GPE, DATE, EVENT, etc.)
        for ent in doc.ents:
            protected_ids.update(range(ent.start, ent.end))

        # 2. Token-level linguistic checks
        for token in doc:
            is_proper_noun = token.pos_ == "PROPN" or token.tag_ in {"NNP", "NNPS"}
            is_acronym = token.text.isupper() and len(token.text) > 1
            is_numeric = token.like_num or token.pos_ == "NUM"
            is_symbol_or_punct = token.is_punct or token.pos_ in {"SYM", "X"}
            is_negation = token.lower_ in self.PROTECTED_WORDS or token.dep_ == "neg"

            if is_proper_noun or is_acronym or is_numeric or is_symbol_or_punct or is_negation:
                protected_ids.add(token.i)

        # 3. Backward scan for full terms paired with parenthesized acronyms
        # Example pattern: "Multiple-choice question answering (MCQA)"
        for i in range(len(doc) - 3):
            # Detect pattern: [Term Token] + "(" + [ACRONYM] + ")"
            if doc[i + 1].text == "(" and doc[i + 2].text.isupper() and doc[i + 3].text == ")":
                curr = i
                # Walk backward to shield all preceding noun phrases, adjectives, and hyphens
                while curr >= 0 and (doc[curr].pos_ in {"NOUN", "PROPN", "ADJ"} or doc[curr].text == "-"):
                    protected_ids.add(curr)
                    curr -= 1

        return protected_ids

    def transform_function(self, text: str) -> str:
        doc = nlp(text)
        protected_ids = self._get_protected_token_indices(doc)

        # Build candidate pool: length >= 4, strictly alphabetic, not in protected index set
        candidate_indices = [
            token.i for token in doc
            if len(token.text) >= 4
            and token.text.isalpha()
            and token.i not in protected_ids
        ]

        # If no words satisfy eligibility criteria, return unchanged input
        if not candidate_indices:
            return text

        # Calculate mutation quota constrained by candidate pool size
        num_mutated = math.ceil(len(candidate_indices) * self.replace_perc)
        selected_ids = self.rand.sample(candidate_indices, min(num_mutated, len(candidate_indices)))

        # Retain exact token formatting and spacing via spaCy's text_with_ws
        tokens = [token.text_with_ws for token in doc]
        mutated_tokens = self.object_transform(selected_ids, tokens)

        return "".join(mutated_tokens)

    def object_transform(self, ids: List[int], text: List[str]) -> List[str]:
        for i in ids:
            full_token = text[i]

            # Separate core alphabetic string from trailing whitespace
            core_word = full_token.rstrip()
            trailing_whitespace = full_token[len(core_word):]

            if len(core_word) > 3:
                # Retry up to 10 times to prevent no-op outcomes on duplicate internal characters
                for _ in range(10):
                    middle_chars = self.rand.sample(core_word[1:-1], len(core_word) - 2)
                    new_word = core_word[0] + "".join(middle_chars) + core_word[-1]

                    if new_word != core_word:
                        text[i] = new_word + trailing_whitespace
                        break

        return text

# original MR-7
class ITRandomiseCharacterOrderInWordKeepingEnds(WordRandomBase):
    def transform_function(self, text):
        tokens = self.tokenise(text)
        ids_at_least_4 = [i for i, token in enumerate(tokens) if len(token) >= 4]
        num_mutated = math.ceil(len(ids_at_least_4) * self.replace_perc)
        ids = self.rand.sample(ids_at_least_4, num_mutated)
        new_text = self.object_transform(ids, tokens)
        return self.join_tokens(new_text)
    
    def object_transform(self, ids: list, text):
        for i in ids:
            if len(text[i]) > 3:
                for _ in range(10): # tries to get different word 10 times
                    new_text = text[i][0] + ''.join(self.rand.sample(text[i][1:-1], len(text[i]) - 2)) + text[i][-1]
                    if new_text != text[i]:
                        text[i] = new_text
                        break
        return text

# MR-9
class ITAddRandomWordAfter(WordRandomBase):
    def get_random_word(self):
        return self.rand.choice(RANDOM_WORDS)
    
    def object_transform(self, ids: list, text):
        out_list = []
        for i, t in enumerate(text):
            out_list.append(t)
            if i in ids:
                random_word = self.get_random_word()
                out_list.append(random_word)
        return out_list


class SentenceRandomBase(ObjectRandomBase):
    def tokenise(self, text):
        return sent_tokenize(text)
    
    def join_tokens(self, tokens):
        return ' '.join(tokens)

# not used
class ITDeleteSentences(SentenceRandomBase):
    def object_transform(self, ids: list, text: list):
        return [sentence for i, sentence in enumerate(text) if i not in ids]

# not used
class ITReplaceSentences(SentenceRandomBase):
    def __init__(self, transform_indices=[[0]], replace_perc=0.1, rand_seed=42):
        super().__init__(transform_indices, replace_perc, rand_seed)
        self.data = RANDOM_SENTENCES

    def object_transform(self, ids: list, text: list):
        dummy_sentence = self.rand.choice(self.data)
        for i in ids:
            text[i] = dummy_sentence
        return text

# shared spaCy model for the dependency-parse-based transforms below (MR-136, MR-149)
_SPACY_NLP = None
def get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP

# MR-136 augmenter (used via ITNlpaug, augment_type='passive_active')
class PassiveActiveAugmenter:
    _PRONOUN_SUBJ_TO_OBJ = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them", "who": "whom"}
    _PRONOUN_OBJ_TO_SUBJ = {v: k for k, v in _PRONOUN_SUBJ_TO_OBJ.items()}

    def __init__(self):
        self.nlp = get_spacy_nlp()

    def augment(self, text):
        doc = self.nlp(text)
        return ' '.join(self._convert_sentence(sent) for sent in doc.sents)

    def _convert_sentence(self, sent):
        root = next((tok for tok in sent if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX")), None)
        if root is None:
            return sent.text
        return self._active_to_passive(sent, root) or self._passive_to_active(sent, root) or sent.text

    def _is_plural(self, token):
        return token.tag_ in ("NNS", "NNPS")

    def _recase(self, span, capitalize_first):
        tokens = list(span)
        if len(tokens) == 1 and tokens[0].pos_ == "PRON":
            word = tokens[0].text.lower()
            text = self._PRONOUN_SUBJ_TO_OBJ.get(word) or self._PRONOUN_OBJ_TO_SUBJ.get(word) or tokens[0].text
        else:
            text = span.text
            if tokens[0].pos_ != "PROPN" and text:
                text = text[0].lower() + text[1:]
        if capitalize_first and text:
            text = text[0].upper() + text[1:]
        return text

    def _be_form(self, tag, plural):
        forms = pyinflect.getInflection("be", tag)
        if not forms:
            return None
        return forms[1] if (plural and len(forms) > 1) else forms[0]

    def _end_punct(self, sent):
        return sent[-1].text if sent[-1].pos_ == "PUNCT" else "."

    # e.g. "Alex discovered penicillin." -> "Penicillin was discovered by Alex."
    def _active_to_passive(self, sent, root):
        if root.tag_ not in ("VBD", "VBZ", "VBP"):
            return None
        subj = next((c for c in root.children if c.dep_ == "nsubj"), None)
        dobj = next((c for c in root.children if c.dep_ == "dobj"), None)
        if subj is None or dobj is None:
            return None
        if any(c.dep_ in ("aux", "auxpass", "neg") for c in root.children):
            return None  # skip modals/negation/already-passive constructs we don't model

        participle = pyinflect.getInflection(root.lemma_, "VBN")
        aux = self._be_form(root.tag_, self._is_plural(dobj))
        if not participle or aux is None:
            return None

        subj_span = sent.doc[subj.left_edge.i: subj.right_edge.i + 1]
        dobj_span = sent.doc[dobj.left_edge.i: dobj.right_edge.i + 1]
        new_subject = self._recase(dobj_span, capitalize_first=True)
        new_object = self._recase(subj_span, capitalize_first=False)
        return f"{new_subject} {aux} {participle[0]} by {new_object}{self._end_punct(sent)}"

    # e.g. "Penicillin was discovered by Alex." -> "Alex discovered penicillin."
    def _passive_to_active(self, sent, root):
        if root.tag_ != "VBN":
            return None
        nsubjpass = next((c for c in root.children if c.dep_ == "nsubjpass"), None)
        auxpass = next((c for c in root.children if c.dep_ == "auxpass"), None)
        agent_prep = next((c for c in root.children if c.dep_ == "agent"), None)
        if nsubjpass is None or auxpass is None or agent_prep is None:
            return None
        pobj = next((c for c in agent_prep.children if c.dep_ == "pobj"), None)
        if pobj is None or any(c.dep_ in ("aux", "neg") for c in root.children):
            return None

        if auxpass.tag_ == "VBD":
            verb_tag = "VBD"
        elif auxpass.text.lower() == "am" or auxpass.tag_ == "VBP":
            verb_tag = "VBP"
        else:
            verb_tag = "VBP" if self._is_plural(pobj) else "VBZ"

        verb_form = pyinflect.getInflection(root.lemma_, verb_tag)
        if not verb_form:
            return None

        nsubjpass_span = sent.doc[nsubjpass.left_edge.i: nsubjpass.right_edge.i + 1]
        pobj_span = sent.doc[pobj.left_edge.i: pobj.right_edge.i + 1]
        new_subject = self._recase(pobj_span, capitalize_first=True)
        new_object = self._recase(nsubjpass_span, capitalize_first=False)
        return f"{new_subject} {verb_form[0]} {new_object}{self._end_punct(sent)}"


# MR-149
class ITSingularPlural(SingleInputTransformer):
    def singular_plural(self, text):
        nlp = get_spacy_nlp()
        doc = nlp(text)
        return ' '.join(self._convert_sentence(sent) for sent in doc.sents)

    def _convert_sentence(self, sent):
        replacements = {}  # token.i -> replacement text, or None to drop the token
        insertions = {}    # token.i -> text to insert before this token
        verb_targets = {}  # governing verb token.i -> new subject number ('singular'/'plural')

        for tok in sent:
            if tok.pos_ != "NOUN" or tok.tag_ not in ("NN", "NNS"):
                continue
            if any(c.dep_ == "nummod" for c in tok.children):
                continue  # skip numeral-modified nouns (e.g. "three boats") - can't singularise sensibly

            is_plural = tok.tag_ == "NNS"
            forms = pyinflect.getInflection(tok.lemma_, "NN" if is_plural else "NNS")
            if not forms:
                continue
            replacements[tok.i] = forms[0]

            # match by tag rather than dep_=='det' since the parser sometimes mislabels
            # determiners on unusual sentences (e.g. temporal "this")
            det = next((c for c in tok.children if c.tag_ == "DT"), None)
            if is_plural:
                # plural -> singular: fix demonstratives, or add an indefinite article if bare
                if det is not None:
                    dtext = det.text.lower()
                    if dtext == "these":
                        replacements[det.i] = "this"
                    elif dtext == "those":
                        replacements[det.i] = "that"
                else:
                    insertions[tok.i] = "an" if forms[0][0].lower() in "aeiou" else "a"
            else:
                # singular -> plural: drop the indefinite article, fix demonstratives
                if det is not None:
                    dtext = det.text.lower()
                    if dtext in ("a", "an"):
                        replacements[det.i] = None
                    elif dtext == "this":
                        replacements[det.i] = "these"
                    elif dtext == "that":
                        replacements[det.i] = "those"

            if tok.dep_ in ("nsubj", "nsubjpass"):
                verb_targets[tok.head.i] = "singular" if is_plural else "plural"

        # keep the governing verb (and any 'be' aux/copula) agreeing with the new subject number
        for verb_i, new_number in verb_targets.items():
            verb = sent.doc[verb_i]
            for vt in [verb] + [c for c in verb.children if c.dep_ in ("aux", "auxpass")]:
                if vt.lemma_ == "be" or vt.tag_ in ("VBZ", "VBP"):
                    forms = pyinflect.getInflection(vt.lemma_, "VBZ" if new_number == "singular" else "VBP")
                    if forms:
                        replacements[vt.i] = forms[0]

        out = []
        for tok in sent:
            insertion = insertions.get(tok.i)
            if insertion is not None:
                out.append(insertion)
                out.append(' ')
            if tok.i in replacements:
                new_text = replacements[tok.i]
                if new_text is None:
                    continue  # dropped (e.g. removed "a"/"an"); also drops its trailing whitespace
                out.append(new_text)
            else:
                out.append(tok.text)
            out.append(tok.whitespace_)

        result = ''.join(out).strip()
        return result[0].upper() + result[1:] if result else result

    def input_transformation(self, input: list):
        return self.transform_input(input, self.singular_plural)



# NLPAUG
# MR 126 (keyboard), 127 (spelling), 128 (ocr), 120 (back_translation), 136 (passive_active)

nlpaug_kwargs = {
    'keyboard': {
        'aug_char_p': 0.1,
        'aug_word_p': 0.1,
    },
    'ocr': {
        'aug_char_p': 0.1,
        'aug_word_p': 0.1,
    },
    'spelling': {
        'aug_p': 0.1,
    },
    'random_insert_word': {
        'model_type': 'word2vec',
        'model_path': 'GoogleNews-vectors-negative300.bin',
        'action': 'insert',
    },
    'synonym': {
        'aug_src': 'wordnet',
    },
    'back_translation': {},
    'passive_active': {},
}

initialised_augmenter_map = {
    'keyboard': nac.KeyboardAug(**nlpaug_kwargs['keyboard']),
    'ocr': nac.OcrAug(**nlpaug_kwargs['ocr']),
    'antonym': naw.AntonymAug(),
    #'random_delete_word': naw.RandomWordAug(),
    'random_insert_word': naw.WordEmbsAug(**nlpaug_kwargs['random_insert_word']),
    'spelling': naw.SpellingAug(**nlpaug_kwargs['spelling']),
    'synonym': naw.SynonymAug(**nlpaug_kwargs['synonym']),
    'back_translation': naw.BackTranslationAug(**nlpaug_kwargs['back_translation']),
    'passive_active': PassiveActiveAugmenter(**nlpaug_kwargs['passive_active']),
}

class ITNlpaug(SingleInputTransformer):
    def __init__(self, augment_type: str, transform_indices=[[0]], **augmenter_kwargs):
        super().__init__(transform_indices)
        self.augmenter = self._initialize_augmenter(augment_type, **augmenter_kwargs)

    def _initialize_augmenter(self, augmenter_type, **kwargs):
        augmenter_map = {
            'keyboard': nac.KeyboardAug,
            'ocr': nac.OcrAug,
            # 'random_char': nac.RandomCharAug,
            'antonym': naw.AntonymAug,
            # 'contextual_word_embs': naw.ContextualWordEmbsAug,
            # 'random_word': naw.RandomWordAug,
            'random_insert_word': naw.WordEmbsAug,
            # 'random_delete_word': naw.RandomWordAug,
            'spelling': naw.SpellingAug,
            # 'split': nac.SplitAug,
            'synonym': naw.SynonymAug,
            # 'tfidf': naw.TfIdfAug,
            # 'word_embs': naw.WordEmbsAug,
            'back_translation': naw.BackTranslationAug,
            'passive_active': PassiveActiveAugmenter,
            # 'reserved': naw.ReservedAug,
            # 'contextual_word_embs_sentence': nas.ContextualWordEmbsForSentenceAug,
            # 'abst_summ': nas.AbstSummAug,
            # 'lambada': nas.LambadaAug,
        }

        if augmenter_type not in augmenter_map:
            raise ValueError(f"Unsupported augmenter type: {augmenter_type}")
        
        # return augmenter_map[augmenter_type](**kwargs)
        return initialised_augmenter_map[augmenter_type]

    def nlp_transform(self, input_val):
        augmented_text = self.augmenter.augment(input_val)
        return augmented_text if isinstance(augmented_text, str) else augmented_text[0]
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.nlp_transform)





# KEYWORD-BASED MRS

class GPTKeywordBase(CleanText, GPTRunner, ITBase):
    def get_keywords_gpt(self, input):
        prompt_template = "Identify names, pronouns, country names, occupations, and similar keywords in the following text:\n\"{INPUT_0}\"\nOnly output the list of words, nothing else."
        examples = [
            [["Sarah is an American software engineer. She works for Microsoft."], "Sarah\nAmerican\nsoftware engineer\nMicrosoft"],
            [["My brother will travel to Japan next month to study Japanese."], "brother\nJapan\nJapanese"],
        ]
        return self.run_gpt(input, prompt_template, examples)
    
    def get_keywords(self, input):
        keywords = self.get_keywords_gpt(input)
        keywords_list = re.split(r'[,\n]+\s*', keywords)
        keywords_list_cleaned = [self.clean_text(keyword) for keyword in keywords_list]
        return keywords_list_cleaned
    
    def bind_context_kw(self, context, keywords):
        if isinstance(context, list):
            context = '\n'.join(context)
        return [context, keywords]
    

class ReplaceKeyword(GPTKeywordBase):
    def replace_words(self, text: str, words_from: list[str], words_to: list[str]):
        # return self.replace_words_manual(text, words_from, words_to)
        return self.replace_words_gpt(text, words_from, words_to)

    def replace_words_manual(self, text: str, words_from: list[str], words_to: list[str]):
        for word_from, word_to in zip(words_from, words_to):
            text = re.sub(r'\b' + word_from + r'\b', word_to, text, flags=re.IGNORECASE)
        return text
    
    def replace_words_gpt(self, text: str, words_from: list[str], words_to: list[str]):
        prompt_template = "Look at this text:\n\"{INPUT_0}\"\nReplace words using the following rules:\n{INPUT_1}\nOnly replace any words that are there. Ignore any rules that are not used. Only output the modified text."
        examples = self.get_replace_examples()
        replace_words = '\n'.join(f"{word_from} -> {word_to}" for word_from, word_to in zip(words_from, words_to))
        return self.run_gpt([text, replace_words], prompt_template, examples)
    
    def get_replace_examples(self):
        pass


# 137 - CATEGORY
class ITReplaceKeywordCategory(ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "Sarah is an American software engineer. She works for Microsoft.", 
            "software engineer -> farmer\nSarah -> John\nAmerican -> Swedish\napple -> pear\nMicrosoft -> Nvidea"], 
            "John is a Swedish farmer. He works for Nvidea."],
            [["My brother will travel to Japan next month to study Japanese.", 
            "sweater -> shirt\nJapanese -> Irish\nmonth -> year\nEurope -> Asia\nbrother -> sister\nJapan -> Italy"], 
            "My sister will travel to Italy next year to study Irish."],]

    def get_word_same_category(self, input):
        prompt_template = "Give a word or phrase in the same category as \"{INPUT_0}\"."
        examples = [
            [["Sarah"], "John"],
            [["software engineer"], "farmer"],
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        cleaned_new_word = self.clean_text(new_word)
        return cleaned_new_word

    def input_transformation(self, input: list):
        combined_inputs = '\n\n'.join(input)
        keywords_list = self.get_keywords(combined_inputs)
        category_words = [self.get_word_same_category(keyword) for keyword in keywords_list]
        outputs = [self.replace_words(input_val, keywords_list, category_words) for input_val in input]
        return [outputs]

class ITReplaceKeywordCategoryQA(ITReplaceKeywordCategory):
    def input_transformation(self, input: list):
        keywords_list = self.get_keywords(input[1]) # keywords from question
        category_words = [self.get_word_same_category(keyword) for keyword in keywords_list]
        outputs = [self.replace_words(input_val, keywords_list, category_words) for input_val in input]
        return [outputs]

class ITReplaceKeywordCategoryRE(ITReplaceKeywordCategory):
    def input_transformation(self, input: list):
        keywords = [input[1], input[2]] # keywords are entities
        keywords_list = [self.clean_text(keyword) for keyword in keywords]
        category_words = [self.get_word_same_category(keyword) for keyword in keywords_list]
        context_new = self.replace_words(input[0], keywords_list, category_words)
        output = [context_new, category_words[0], category_words[1]]
        return [output]

# 8 - SYNONYM - OG
class ITReplaceKeywordSynonym_og(SingleInputTransformer, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "Sam walked to the store to buy an apple.", 
            "walked -> travelled\nslowly -> unhurriedly\nbuy -> purchase\nstore -> shop"], 
            "I travelled to the shop to purchase an apple."],
            [["I wonder why clothes are so expensive.", 
            "buy -> purachase\nclothes -> garments\nexpensive -> pricy\nwhy -> for what reason\nfull -> complete"], 
            "I wonder for what reason garments are so pricy."],]

    def get_synonym(self, input):
        prompt_template = "Context:\n\"{INPUT_0}\"\nMaking sense in this context, give a synonym for \"{INPUT_1}\". If the word has no synonym, simply output the word itself."
        examples = [
            [["Sam walked to the store to buy an apple.", "walked"], "travelled"],
            [["I wonder why clothes are so expensive.", "expensive"], "pricey"],
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        cleaned_new_word = self.clean_text(new_word)
        return cleaned_new_word
    
    def replace_synonym(self, input):
        keywords = self.get_keywords(input)
        synonym_words = [self.get_synonym(self.bind_context_kw(input, keyword)) for keyword in keywords]
        return self.replace_words(input, keywords, synonym_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_synonym)

# 8 - SYNONYM - prompt_template
class ITReplaceKeywordSynonym(SingleInputTransformer, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "Sam walked to the store to buy an apple.", 
            "walked -> travelled\nslowly -> unhurriedly\nbuy -> purchase\nstore -> shop"], 
            "I travelled to the shop to purchase an apple."],
            [["I wonder why clothes are so expensive.", 
            "buy -> purachase\nclothes -> garments\nexpensive -> pricy\nwhy -> for what reason\nfull -> complete"], 
            "I wonder for what reason garments are so pricy."],]

    def get_synonym(self, input):
        prompt_template = "Context:\n\"{INPUT_0}\"\nMaking sense in this context, give a synonym for \"{INPUT_1}\". Don’t change technical terms, proper nouns, model names, or acronyms (e.g., MCQA, MMT, AI), leave it unchanged. Preserve compound hyphens (e.g., 'Multiple-choice') and exact casing. If replacing the word changes the context, keep the original word unchanged. Do not replace words with generic/unrelated words. Maintain strict grammatical correctness. If the word has no synonym, simply output the word itself. Maintain exact Part-of-Speech, tense, and singular/plural forms (e.g., Noun -> Noun, Plural -> Plural)."
        examples = [
            [["Extensive experiments demonstrate continuous improvements across models.", "experiments"], "tests"],
            [["I wonder why clothes are so expensive in this store.", "expensive"], "pricey"],
            [["We evaluate on multi-source settings.", "multi-source"], "multi-source"]
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        #cleaned_new_word = self.clean_text(new_word)
        #return cleaned_new_word
        return new_word
   
    #def replace_synonym(self, input):
    #    keywords = self.get_keywords(input)
    #    synonym_words = [self.get_synonym(self.bind_context_kw(input, keyword)) for keyword in keywords]
    #    return self.replace_words(input, keywords, synonym_words)

    def replace_synonym(self, input):
        keywords = self.get_keywords(input)
        synonym_words = []
        
        # Set Cosine Similarity threshold to filter out semantic drifts/out-of-context predictions
        SIMILARITY_THRESHOLD = 0.75 
        
        for keyword in keywords:
            # Generate target synonym using the model with context
            candidate = self.get_synonym(self.bind_context_kw(input, keyword))
            
            # Clean LLM output (strip quotes, spaces, and extra newlines)
            if hasattr(self, 'clean_text'):
                candidate = self.clean_text(candidate)
            else:
                candidate = candidate.strip(' "\'\n\r')
            
            # Case 1: If candidate is empty or identical to original keyword, keep original
            if not candidate or candidate == keyword:
                synonym_words.append(keyword)
                continue
            
            # Case 2: Calculate Cosine Similarity between keyword and generated candidate
            # (Replace 'compute_cosine_sim' with your actual similarity function/method)
            sim_score = self.compute_cosine_sim(keyword, candidate)
            
            # Case 3: Filter by threshold
            if sim_score >= SIMILARITY_THRESHOLD:
                synonym_words.append(candidate) # Accept valid candidate
            else:
                # Fallback: Reject candidate if semantic drift occurs (e.g., sim < 0.75)
                synonym_words.append(keyword)
                
        return self.replace_words(input, keywords, synonym_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_synonym)

class ReplaceKeywordDifferenceRE(ReplaceKeyword):
    def get_keywords_gpt(self, text, input_1, input_2):
        prompt_template = "Here are two input words:\n\"{INPUT_1}\"\n\"{INPUT_2}\"\n\nIdentify names, pronouns, country names, occupations, and similar keywords in the following text that is associated with these words:\n\"{INPUT_0}\"\nOnly output the list of words, nothing else."
        examples = [
            [["Sarah is an American software engineer. She works for Microsoft.", "Sarah", "Microsoft"], "American\nsoftware engineer\nworks for"],
            [["My brother will travel to Japan next month to study Japanese.", "brother", "Japan"], "travel\nnext month\nstudy\nJapanese"],
        ]
        return self.run_gpt([text, input_1, input_2], prompt_template, examples)
    
    def get_keywords(self, text, input_1, input_2):
        keywords = self.get_keywords_gpt(text, input_1, input_2)
        keywords_list = re.split(r'[,\n]+\s*', keywords)
        keywords_list_cleaned = [self.clean_text(keyword) for keyword in keywords_list]
        cleaned_inputs = [self.clean_text(input_1), self.clean_text(input_2)]
        keywords_out = [keyword for keyword in keywords_list_cleaned if keyword not in cleaned_inputs] # remove entities if appear in keywords
        return keywords_out

# 10 - ANTONYM -OG
class ITReplaceKeywordAntonym_og(SingleInputTransformer, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "She walked to the store to buy an apple.", 
            "walked -> ran\nslowly -> quickly\nbuy -> sell\nstore -> home\nshe -> he"], 
            "He ran to the home to sell an apple."],
            [["In 1993, I broke my arm while cleaning my electric car.", 
            "noisy -> silent\nmy -> your\nfull -> empty\nelectric -> petrol\ncleaning -> dirtying\nbroke -> fixed"], 
            "In 1993, I fixed your arm while dirtying your petrol car."],]

    def get_antonym(self, input):
        # prompt_template = "Context:\n\"{INPUT_0}\"\nMaking sense in this context, give an antonym for \"{INPUT_1}\". If the word has no antonym, simply output the word itself."
        prompt_template = "You are given a context and a word. Produce an antonym of the word. Make sure the antonym makes sense in the context. If the word has no antonym, simply output the word itself. \n<context>{INPUT_0}</context>\n<word>{INPUT_1}</word>"
        examples = [
            [["She walked to the store to buy an apple.", "buy"], "sell"],
            [["In 1993, I broke my arm while cleaning my electric car.", "electric"], "petrol"],
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        cleaned_new_word = self.clean_text(new_word)
        return cleaned_new_word
    
    def replace_antonym(self, input):
        keywords = self.get_keywords(input)
        antonym_words = [self.get_antonym(self.bind_context_kw(input, keyword)) for keyword in keywords]
        return self.replace_words(input, keywords, antonym_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_antonym)

# 10 - ANTONYM - prompt_template
class ITReplaceKeywordAntonym(SingleInputTransformer, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "She walked to the store to buy an apple.", 
            "walked -> ran\nslowly -> quickly\nbuy -> sell\nstore -> home\nshe -> he"], 
            "He ran to the home to sell an apple."],
            [["In 1993, I broke my arm while cleaning my electric car.", 
            "noisy -> silent\nmy -> your\nfull -> empty\nelectric -> petrol\ncleaning -> dirtying\nbroke -> fixed"], 
            "In 1993, I fixed your arm while dirtying your petrol car."],]

    def get_antonym(self, input):
        # prompt_template = "Context:\n\"{INPUT_0}\"\nMaking sense in this context, give an antonym for \"{INPUT_1}\". If the word has no antonym, simply output the word itself."
        prompt_template = "Replace eligible words in the following text with antonyms to change its meaning:\\n\"{INPUT_0}\"\\nRules:\\n1. NEVER replace technical terms, proper nouns, model names, or acronyms.\\n2. If a word lacks a clear, logical antonym in context, leave it unchanged. Do not replace words randomly or with generic terms.\\n3. Maintain grammatical correctness and logical structure.\\nOnly output the changed text, nothing else." 
        examples = [
            [["Is Scott and Sid based on a true story?"], "Is Scott and Sid based on a false story?"],
            [["Most existing MCQA datasets are small in size, which increases the difficulty of model learning."], "Most existing MCQA datasets are large in size, which increases the difficulty of model learning."
      ],
        [["The proposed MMT framework is independent of backbone language models."], "The proposed MMT framework is dependent on backbone language models."],
            [["Continuous improvements can be achieved on different backbone networks."], "Continuous declines can be achieved on different backbone networks."]
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        cleaned_new_word = self.clean_text(new_word)
        return cleaned_new_word
    
    def replace_antonym(self, input):
        keywords = self.get_keywords(input)
        antonym_words = [self.get_antonym(self.bind_context_kw(input, keyword)) for keyword in keywords]
        return self.replace_words(input, keywords, antonym_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_antonym)

class ITReplaceKeywordAntonymQA(ITReplaceKeywordAntonym):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[1]) # keywords from question
        antonym_words = [self.get_antonym(self.bind_context_kw(input[1], keyword)) for keyword in keywords]
        output_c = self.replace_words(input[0], keywords, antonym_words)
        output_q = self.replace_words(input[1], keywords, antonym_words)
        return [[output_c, input[1]], [input[0], output_q]] # either, not both

class ITReplaceKeywordAntonymRE(ReplaceKeywordDifferenceRE, ITReplaceKeywordAntonym):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[0], input[1], input[2])
        antonym_words = [self.get_antonym(self.bind_context_kw(input[0], keyword)) for keyword in keywords]
        output = self.replace_words(input[0], keywords, antonym_words)
        return [[output, input[1], input[2]]]
    


# 25 - RANDOM
class ITReplaceKeywordRandom(SingleInputRandomBase, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "He walked to the store to buy an apple.", 
            "walked -> cut\nslowly -> break\nbuy -> run\napple -> play\nstore -> fall"], 
            "He cut to the fall to run an play."],
            [["Sarah is an American software engineer. She works for Microsoft.", 
            "software engineer -> carry\nSarah -> give\nAmerican -> light\nsandwich -> clear\nMicrosoft -> call"], 
            "Give is a light carry. She works for call."],]

    def get_random_words(self, n):
        return list(self.rand.sample(RANDOM_WORDS, n))
    
    def replace_random_word(self, input):
        keywords = self.get_keywords(input)
        random_words = self.get_random_words(len(keywords))
        return self.replace_words(input, keywords, random_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_random_word)

class ITReplaceKeywordRandomQA(ITReplaceKeywordRandom):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[1]) # keywords from question
        random_words_c = self.get_random_words(len(keywords))
        random_words_q = self.get_random_words(len(keywords))
        output_c = self.replace_words(input[0], keywords, random_words_c)
        output_q = self.replace_words(input[1], keywords, random_words_q)
        return [[output_c, input[1]], [input[0], output_q], [output_c, output_q]] # all combinations

class ITReplaceKeywordRandomRE(ReplaceKeywordDifferenceRE, ITReplaceKeywordRandom):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[0], input[1], input[2])
        random_words = self.get_random_words(len(keywords))
        output = self.replace_words(input[0], keywords, random_words)
        return [[output, input[1], input[2]]]


# 34 - REMOVE
class ITRemoveKeyword(SingleInputTransformer, GPTKeywordBase):
    def remove_keywords(self, input_val, keywords):
        # return self.remove_keywords_manual(input_val, keywords)
        return self.remove_keywords_gpt(input_val, keywords)
    
    def remove_keywords_manual(self, input_val, keywords):
        new_text = input_val
        for keyword in keywords:
            new_text = re.sub(r'\b' + keyword + r'\b', '', new_text, flags=re.IGNORECASE)
        return new_text
    
    def get_gpt_prompt(self):
        prompt_template = "Look at this text:\n\"{INPUT_0}\"\nRemove the following words:\n{INPUT_1}\nOnly remove the words. Only output the modified text."
        examples = [
            [["Sarah is an American software engineer. She works for Microsoft.", "Sarah\nAmerica\nsoftware engineer\nMicrosoft"], "is an. She works for."],
            [["My brothers will travel to Japan next month to study Japanese.", "brother\nJapan"], "My will travel to next to study."],
        ]
        return prompt_template, examples
    
    def remove_keywords_gpt(self, input_val, keywords):
        prompt_template, examples = self.get_gpt_prompt()
        return self.run_gpt([input_val, '\n'.join(keywords)], prompt_template, examples)

    def get_and_remove_keywords(self, input):
        keywords = self.get_keywords(input)
        return self.remove_keywords(input, keywords)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.get_and_remove_keywords)

# MR-51
class ITParaphrasing(SingleInputTransformer):
    """
    MR-51: Offline paraphrase transformer using context-preserving WordNet synonym substitution.
    Guards Named Entities, Proper Nouns, and uppercase acronyms to preserve ground truth.
    """
    def __init__(self, transform_indices=[[0]], aug_p=0.2, nlp=None):
        super().__init__(transform_indices)
        self.nlp = nlp if nlp else spacy.load("en_core_web_sm")
        self.aug = naw.SynonymAug(aug_src="wordnet", aug_p=aug_p)

    def paraphrase_transform(self, input_val: str) -> str:
        if not input_val.strip():
            return input_val

        doc = self.nlp(input_val.strip())
        
        # Protect entities, acronyms, and numbers
        protected_words = set()
        for ent in doc.ents:
            for token in ent:
                protected_words.add(token.text)
        for token in doc:
            if token.like_num or (token.text.isupper() and len(token.text) > 1):
                protected_words.add(token.text)

        # Configure stopwords so nlpaug skips critical keywords
        self.aug.stopwords = list(protected_words)
        
        augmented = self.aug.augment(input_val)
        return augmented[0] if isinstance(augmented, list) else augmented

    def input_transformation(self, input: list):
        return self.transform_input(input, self.paraphrase_transform)

class ITRemoveKeywordSentence(ITRemoveKeyword):
    def remove_keywords_gpt(self, input_val, keywords):
        prompt_template, examples = self.get_gpt_prompt()

        input_sentences = sent_tokenize(input_val)
        keyword_string = '\n'.join(keywords)

        output = [self.run_gpt([sentence, keyword_string], prompt_template, examples) for sentence in input_sentences]
        return ' '.join(output)
    
class ITRemoveKeywordQA(ITRemoveKeyword):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[1]) # keywords from question
        output_c = self.remove_keywords(input[0], keywords)
        output_q = self.remove_keywords(input[1], keywords)
        return [[output_c, input[1]], [input[0], output_q], [output_c, output_q]] # all combinations

class ITRemoveKeywordQASentence(ITRemoveKeywordQA, ITRemoveKeywordSentence):
    pass

class ITRemoveKeywordRE(ReplaceKeywordDifferenceRE, ITRemoveKeyword):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[0], input[1], input[2])
        output = self.remove_keywords(input[0], keywords)
        return [[output, input[1], input[2]]]
    
class ITRemoveKeywordRESentence(ITRemoveKeywordRE, ITRemoveKeywordSentence):
    pass

# MR-136
class ITPassiveActiveVoice(SingleInputTransformer):
    """
    MR-136: Active/Passive Voice alternation without an LLM.
    Uses SpaCy dependency parsing to deterministically convert active sentences 
    to passive voice, and vice-versa, preserving semantic equivalence.
    """
    def __init__(self, transform_indices=[[0]], nlp=None):
        super().__init__(transform_indices)
        self.nlp = nlp if nlp else spacy.load("en_core_web_sm")

    def _convert_passive_to_active(self, doc) -> str:
        agent_phrase = None
        subjpass = None
        root_verb = None
        auxpass = None

        for token in doc:
            if token.dep_ == "nsubjpass":
                subjpass = token
            elif token.dep_ == "auxpass":
                auxpass = token
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                root_verb = token
            elif token.dep_ == "agent":
                # Find the object of preposition 'by'
                for child in token.children:
                    if child.dep_ == "pobj":
                        agent_phrase = "".join([w.text_with_ws for w in child.subtree]).strip()

        if agent_phrase and subjpass and root_verb:
            # Subject subtree
            subj_text = "".join([w.text_with_ws for w in subjpass.subtree]).strip()
            
            # Simple past / active verb resolution
            active_verb = root_verb.lemma_
            if auxpass and auxpass.text.lower() in {"was", "were", "had", "been"}:
                # Basic regular past inflection heuristic fallback
                active_verb = root_verb.lemma_ + ("d" if root_verb.lemma_.endswith("e") else "ed")
                if root_verb.text.lower() not in {"discovered", "created", "eaten"}:
                    active_verb = root_verb.text

            punct = doc[-1].text if doc[-1].is_punct else ""
            return f"{agent_phrase.capitalize()} {active_verb} {subj_text.lower()}{punct}"

        return doc.text

    def _convert_active_to_passive(self, doc) -> str:
        subj = None
        dobj = None
        root_verb = None

        for token in doc:
            if token.dep_ == "nsubj":
                subj = token
            elif token.dep_ == "dobj":
                dobj = token
            elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                root_verb = token

        if subj and dobj and root_verb:
            subj_text = "".join([w.text_with_ws for w in subj.subtree]).strip()
            dobj_text = "".join([w.text_with_ws for w in dobj.subtree]).strip()
            
            # Auxiliary verb selection based on tense and number
            is_past = root_verb.tag_ in {"VBD", "VBN"}
            is_plural = dobj.tag_ in {"NNS", "NNPS"}
            
            if is_past:
                aux = "were" if is_plural else "was"
            else:
                aux = "are" if is_plural else "is"

            # Past participle
            verb_participle = root_verb.text if root_verb.tag_ == "VBN" else (
                root_verb.lemma_ + ("d" if root_verb.lemma_.endswith("e") else "ed")
            )
            if root_verb.lemma_ == "eat":
                verb_participle = "eaten"

            punct = doc[-1].text if doc[-1].is_punct else ""
            return f"{dobj_text.capitalize()} {aux} {verb_participle} by {subj_text.lower()}{punct}"

        return doc.text

    def voice_transform(self, input_val: str) -> str:
        if not input_val.strip():
            return input_val

        doc = self.nlp(input_val.strip())
        is_passive = any(tok.dep_ == "auxpass" or tok.dep_ == "nsubjpass" for tok in doc)

        if is_passive:
            return self._convert_passive_to_active(doc)
        return self._convert_active_to_passive(doc)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.voice_transform)

# MR-149
class ITSingularPlural(SingleInputTransformer):
    """
    MR-149: Singular <-> Plural alternation without an LLM.
    Uses SpaCy POS and morphology tagging to toggle noun numbers
    and balance subject-verb agreement while preserving named entities.
    """
    def __init__(self, transform_indices=[[0]], nlp=None):
        super().__init__(transform_indices)
        self.nlp = nlp if nlp else spacy.load("en_core_web_sm")

    def _inflect_noun(self, token) -> str:
        """
        Converts singular noun to plural, and plural noun to singular.
        """
        lemma = token.lemma_.lower()
        word = token.text

        # Plural -> Singular
        if token.tag_ in {"NNS", "NNPS"}:
            if word.isupper():
                return lemma.upper()
            if word.istitle():
                return lemma.capitalize()
            return lemma

        # Singular -> Plural
        if token.tag_ in {"NN", "NNP"}:
            # Common irregular plurals
            irregulars = {
                "cabbage": "cabbages",
                "child": "children",
                "person": "people",
                "man": "men",
                "woman": "women",
                "tooth": "teeth",
                "foot": "feet",
                "mouse": "mice",
                "datum": "data",
            }
            if lemma in irregulars:
                plural = irregulars[lemma]
            elif word.endswith(("s", "x", "z", "ch", "sh")):
                plural = word + "es"
            elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiouAEIOU":
                plural = word[:-1] + "ies"
            else:
                plural = word + "s"

            if word.isupper():
                return plural.upper()
            if word.istitle():
                return plural.capitalize()
            return plural

        return word

    def number_transform(self, input_val: str) -> str:
        if not input_val.strip():
            return input_val

        doc = self.nlp(input_val.strip())
        output_tokens = []
        skip_next = False

        # Gather tokens to avoid mutating Named Entities, Acronyms, or Numbers
        protected_indices = set()
        for ent in doc.ents:
            for idx in range(ent.start, ent.end):
                protected_indices.add(idx)

        for i, token in enumerate(doc):
            if skip_next:
                skip_next = False
                continue

            # 1. Skip protected tokens (Named Entities, all-caps acronyms > 1 char, numbers)
            if i in protected_indices or token.like_num or (token.text.isupper() and len(token.text) > 1):
                output_tokens.append(token.text_with_ws)
                continue

            # 2. Handle Indefinite Determiners preceding Singular Nouns ("a cabbage" -> "cabbages")
            if token.lower_ in {"a", "an"} and i + 1 < len(doc) and doc[i + 1].tag_ == "NN":
                # Drop "a/an" when the following noun becomes plural
                continue

            # 3. Toggle Noun Number (NN <-> NNS)
            if token.pos_ == "NOUN" and token.tag_ in {"NN", "NNS"}:
                new_noun = self._inflect_noun(token)
                output_tokens.append(new_noun + token.whitespace_)
                continue

            # 4. Synchronize Auxiliary/Be-Verb agreement
            lower_txt = token.text.lower()
            verb_map = {
                "is": "are", "are": "is",
                "was": "were", "were": "was",
                "has": "have", "have": "has"
            }
            if lower_txt in verb_map and token.pos_ in {"AUX", "VERB"}:
                mapped_verb = verb_map[lower_txt]
                if token.text.istitle():
                    mapped_verb = mapped_verb.capitalize()
                output_tokens.append(mapped_verb + token.whitespace_)
                continue

            output_tokens.append(token.text_with_ws)

        return "".join(output_tokens)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.number_transform)

# 152 - NEGATE
class ITNegateSpacy(SingleInputTransformer, ITBase):
    def get_negated(self, input: list) -> str | None:
        output = [self.negate_sentence(i) for i in input]
        print("BEFORE:", output)
        return "".join(output)

    def negate_sentence(self, text: str) -> str:
        doc = nlp(text)
        tokens = [t.text_with_ws for t in doc]

        for token in doc:
            # Skip if already negated
            if any(child.dep_ == "neg" for child in token.children):
                return text
            # ---------- CASE 1 ----------
            # Auxiliary verb (is, are, was, has, have, will, can...)
            if token.pos_ in ("AUX", "VERB"):
                # Ignore "do" auxiliaries
                if token.lemma_ == "do":
                    continue

                # Only finite auxiliaries
                if token.dep_ == "aux" or token.tag_ in (
                    "VBZ",
                    "VBP",
                    "VBD",
                    "MD"
                ):
                    # be / have / modal
                    if token.lemma_ in (
                        "be",
                        "have",
                        "will",
                        "can",
                        "could",
                        "should",
                        "would",
                        "may",
                        "might",
                        "must",
                        "shall"
                    ):
                        tokens[token.i] = token.text + " not" + token.whitespace_
                        return "".join(tokens)

            # ---------- CASE 2 ----------
            # Main verb without auxiliary
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                has_aux = any(c.dep_ == "aux" for c in token.children)
                if has_aux:
                    continue

                lemma = token.lemma_

                if token.tag_ == "VBZ":
                    replacement = f"does not {lemma}"
                elif token.tag_ == "VBD":
                    replacement = f"did not {lemma}"
                else:
                    replacement = f"do not {lemma}"
                tokens[token.i] = replacement + token.whitespace_

                return "".join(tokens)
        return text

import spacy

# MR-155
class ITTenseChange(SingleInputTransformer):
    """
    MR-155: Tense change transformation without an LLM.
    Uses SpaCy POS tagging and verb morphology to toggle past and present tenses
    while maintaining Named Entities and acronym integrity.
    """
    def __init__(self, transform_indices=[[0]], nlp=None):
        super().__init__(transform_indices)
        self.nlp = nlp if nlp else spacy.load("en_core_web_sm")

    def _shift_verb_tense(self, token) -> str:
        tag = token.tag_
        lemma = token.lemma_.lower()
        word = token.text

        # Auxiliary mappings
        aux_to_past = {"is": "was", "are": "were", "am": "was", "has": "had", "have": "had", "do": "did", "does": "did", "will": "would", "can": "could"}
        aux_to_pres = {"was": "is", "were": "are", "had": "has", "did": "does", "would": "will", "could": "can"}

        # Irregular lexical verbs
        irregulars_past = {"eat": "ate", "see": "saw", "go": "went", "take": "took", "find": "found", "make": "made", "get": "got", "know": "knew"}
        irregulars_pres = {v: k for k, v in irregulars_past.items()}

        # 1. Past -> Present
        if tag == "VBD":
            if word.lower() in aux_to_pres:
                res = aux_to_pres[word.lower()]
            elif word.lower() in irregulars_pres:
                res = irregulars_pres[word.lower()]
            else:
                res = lemma
            return res.capitalize() if word.istitle() else res

        # 2. Present -> Past
        if tag in {"VBP", "VBZ"}:
            if word.lower() in aux_to_past:
                res = aux_to_past[word.lower()]
            elif lemma in irregulars_past:
                res = irregulars_past[lemma]
            else:
                res = lemma + ("d" if lemma.endswith("e") else "ed")
            return res.capitalize() if word.istitle() else res

        return word

    def tense_transform(self, input_val: str) -> str:
        if not input_val.strip():
            return input_val

        doc = self.nlp(input_val.strip())
        output_tokens = []

        # Avoid modifying verbs inside Named Entities
        ent_indices = {i for ent in doc.ents for i in range(ent.start, ent.end)}

        for i, token in enumerate(doc):
            if i in ent_indices or token.text.isupper() and len(token.text) > 1:
                output_tokens.append(token.text_with_ws)
                continue

            if token.pos_ in {"VERB", "AUX"} and token.tag_ in {"VBD", "VBP", "VBZ"}:
                new_verb = self._shift_verb_tense(token)
                output_tokens.append(new_verb + token.whitespace_)
            else:
                output_tokens.append(token.text_with_ws)

        return "".join(output_tokens)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.tense_transform)

class ITNegate(GPTRunner, SingleInputTransformer, ITBase):
    def get_prompt(self):
        prompt_template = "Negate the following text with minimal change:\n\"{INPUT_0}\"\nOnly output the changed text, nothing else."
        examples = [
            [["Who was the leader of Norway?"], "Who wasn't the leader of Norway?"],
            [["The President ate a cabbage."], "The President didn't eat a cabbage."]
        ]
        return prompt_template, examples
    
    def get_negated(self, input: list):
        prompt_template, examples = self.get_prompt()
        return self.run_gpt(input, prompt_template, examples)
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.get_negated)

class ITNegateQA(ITNegateSpacy):
    """ # Use LLM
    def get_negated_context(self, input: list):
        prompt_template = "Given this question:\n\"{INPUT_1}\"\n\nNegate the following text with minimal change such that the information relavent to the question is the opposite:\n\"{INPUT_0}\"\nOnly output the changed text, nothing else."
        examples = [
            [["The leader of Norway was Jonas Gahr Store, son of a wealthy ship broker.", "Who was the leader of Norway?"], "The leader of Norway wasn't Jonas Gahr Store, son of a wealthy ship broker."],
            [["She went to the shops, ate a cabbage, and returned home with a basketball.", "What did she eat?"], "She went to the shops, didn't eat a cabbage, and returned home with a basketball."]
        ]
        return self.run_gpt(input, prompt_template, examples)
    """
    
    def input_transformation(self, input: list):
        # input: [context, question]
        negated_context = self.get_negated(input) # self.get_negated_context(input) # Use llm
        negated_question = self.get_negated([input[1]])
        return [[negated_context, input[1]], [input[0], negated_question]]

class ITNegateRE(ITNegate):
    def get_negated_re(self, input: list):
        prompt_template = "Negate the following text with minimal change such that the relationship from \"{INPUT_1}\" to \"{INPUT_2}\" is the opposite:\n\"{INPUT_0}\"\nNegate the text so that the relationship from \"{INPUT_1}\" to \"{INPUT_2}\" is the opposite.\nOnly output the changed text, nothing else."
        examples = [
            [["The leader of Norway was Jonas Gahr Store, son of a wealthy ship broker.", "Jonas", "Norway"], "The leader of Norway wasn't Jonas Gahr Store, son of a wealthy ship broker."],
            [["She went to the shops, ate a cabbage, and returned home with a basketball.", "she", "cabbage"], "She went to the shops, didn't eat a cabbage, and returned home with a basketball."]
        ]
        return self.run_gpt(input, prompt_template, examples)
    
    
