import string
from abc import ABC, abstractmethod

# from contextlib import contextmanager
from typing import List, Optional

# from roar.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
#     get_grapheme_character_set,
#     get_ipa_punctuation_list,
#     validate_locale,
# )
from roar.collections.common.tokenizers.text_to_speech.tokenizer_utils import (
    any_locale_text_preprocessing,
    english_text_preprocessing,
    get_characters_from_range,
)
from roar.utils import logging
from roar.utils.decorators import experimental


class BaseTokenizer(ABC):
    PAD, BLANK, OOV = "<pad>", "<blank>", "<oov>"

    def __init__(
        self, tokens, *, pad=PAD, blank=BLANK, oov=OOV, sep="", add_blank_at=None
    ):
        """Abstract class for creating an arbitrary tokenizer to convert string to list of int tokens.
        Args:
            tokens: List of tokens.
            pad: Pad token as string.
            blank: Blank token as string.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                if None then no blank in labels.
        """
        super().__init__()

        tokens = list(tokens)
        # TODO in general, IDs of pad, sil, blank, and oov are preserved ahead instead of dynamically
        #  assigned according to the number of tokens. The downside of using dynamical assignment leads to different IDs
        #  for each.
        self.pad, tokens = len(tokens), tokens + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, tokens = len(tokens), tokens + [
                blank
            ]  # Reserved for blank from asr-model
        else:
            # use add_blank_at=None only for ASR where blank is added automatically, disable blank here
            self.blank = None

        self.oov, tokens = len(tokens), tokens + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
            self.oov, self.blank = self.blank, self.oov

        self.tokens = tokens
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._token2id = {l: i for i, l in enumerate(tokens)}
        self._id2token = tokens

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(
            self._id2token[t] for t in tokens if t not in self._util_ids
        )


class BaseCharsTokenizer(BaseTokenizer):
    # fmt: off
    # TODO: unify definition of the default PUNCT_LIST and import from ipa_lexicon.py
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on

    def __init__(
        self,
        chars,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=lambda x: x,
    ):
        """Base class for char-based tokenizer.
        Args:
            chars: string that represents all possible characters.
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
        """

        tokens = []
        self.space, tokens = len(tokens), tokens + [" "]  # Space
        tokens.extend(chars)
        if apostrophe:
            tokens.append("'")  # Apostrophe for saving "don't" and "Joe's"

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func

    def encode(self, text):
        """See base class."""
        cs, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        for c in text:
            # Add a whitespace if the current char is a whitespace while the previous char is not a whitespace.
            if c == space and len(cs) > 0 and cs[-1] != space:
                cs.append(c)
            # Add the current char that is an alphanumeric or an apostrophe.
            elif (c.isalnum() or c == "'") and c in tokens:
                cs.append(c)
            # Add a punctuation that has a single char.
            elif (c in self.PUNCT_LIST) and self.punct:
                cs.append(c)
            # Warn about unknown char
            elif c != space:
                logging.warning(
                    f"Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped."
                )

        # Remove trailing spaces
        if cs:
            while cs[-1] == space:
                cs.pop()

        if self.pad_with_space:
            cs = [space] + cs + [space]

        return [self._token2id[p] for p in cs]


class IndicCharsTokenizer(BaseCharsTokenizer):
    __name__ = "IndicCharsTokenizer"
    # fmt: off
    # TODO: unify definition of the default PUNCT_LIST and import from ipa_lexicon.py
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}', 
        '%', '$', '#',
    )
    # fmt: on

    def __init__(
        self,
        chars=None,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        unicode_range=None,
        pad_with_space=False,
        non_default_punct_list=None,
        process_mixed_language_chars=True,
        text_preprocessing_func=any_locale_text_preprocessing,  # TODO: Change this to have indic_nlp_resources' text normalization logic
    ):
        """Base class for char-based tokenizer.
        Args:
            chars: string that represents all possible characters.
            unicode_range: Tuple of unicode range. Only 2 values allowed. Supercedes chars.
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            process_mixed_language_chars: Boolean flag indicating whether or not to process mixed language chars.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
        """
        if chars is None and unicode_range is None:
            raise ValueError("Either chars or unicode_range must be provided.")
        if unicode_range:
            self.in_unicode_range = lambda x: unicode_range[0] <= x <= unicode_range[1]
            chars = get_characters_from_range(*unicode_range)
        else:
            self.in_unicode_range = lambda x: True
            chars = [
                c
                for c in chars
                if c not in string.punctuation
                and c not in self.PUNCT_LIST
                and c.isprintable()
            ]
        self.process_mixed_language_chars = process_mixed_language_chars
        if self.process_mixed_language_chars:
            chars.extend(string.ascii_lowercase)
        chars.extend(map(str, range(10)))
        super().__init__(
            chars=chars,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )

    def encode(self, text):
        """See base class."""
        cs, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        for c in text:
            # Add a whitespace if the current char is a whitespace while the previous char is not a whitespace.
            if c == space and len(cs) > 0 and cs[-1] != space:
                cs.append(c)

            elif ((self.in_unicode_range(c) or c == "'") and c in tokens) or c.isdigit():
                cs.append(c)
            elif (
                self.process_mixed_language_chars
                and c.lower() in string.ascii_lowercase  # for mixed-language chars
            ):
                cs.append(c.lower())
            # Add a punctuation that has a single char.
            elif (c in self.PUNCT_LIST) and self.punct:
                cs.append(c)
            # Warn about unknown char
            elif c != space:
                logging.warning(
                    f"Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped."
                )

        # Remove trailing spaces
        if cs:
            while cs[-1] == space:
                cs.pop()

        if self.pad_with_space:
            cs = [space] + cs + [space]

        return [self._token2id[p] for p in cs]


class TamilCharsTokenizer(IndicCharsTokenizer):
    __name__ = "TamilCharsTokenizer"
    UNICODE_RANGE = ("\u0B80", "\u0BFF")

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        process_mixed_language_chars=True,
        text_preprocessing_func=any_locale_text_preprocessing,
    ):
        """Tamil char-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            process_mixed_language_chars: Boolean flag indicating whether or not to process mixed language chars.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
                Basically, it replaces all non-unicode characters with unicode ones and apply lower() function.
        """
        super().__init__(
            unicode_range=self.UNICODE_RANGE,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            process_mixed_language_chars=process_mixed_language_chars,
            text_preprocessing_func=text_preprocessing_func,
        )


class HindiCharsTokenizer(IndicCharsTokenizer):
    __name__ = "HindiCharsTokenizer"
    UNICODE_RANGE = ("\u0900", "\u097F")

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        process_mixed_language_chars=True,
        text_preprocessing_func=any_locale_text_preprocessing,
    ):
        """Hindi char-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            process_mixed_language_chars: Boolean flag indicating whether or not to process mixed language chars.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
                Basically, it replaces all non-unicode characters with unicode ones and apply lower() function.
        """
        super().__init__(
            unicode_range=self.UNICODE_RANGE,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            process_mixed_language_chars=process_mixed_language_chars,
            text_preprocessing_func=text_preprocessing_func,
        )


class EnglishCharsTokenizer(BaseCharsTokenizer):
    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=english_text_preprocessing,
    ):
        """English char-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones and apply lower() function.
        """
        super().__init__(
            chars=string.ascii_lowercase,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )
