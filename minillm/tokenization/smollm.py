"""Small, self-contained byte-level BPE tokenizer for SmolLM2.

The vocabulary, merge order, added tokens, and chat template come from the
model's local tokenizer files. This module does not delegate custom encoding to
Hugging Face or the Rust ``tokenizers`` package.
"""

from __future__ import annotations

import heapq
import json
from functools import lru_cache
from pathlib import Path

import regex
from jinja2 import Environment

# ByteLevel's GPT-2 pretokenization expression.
_BYTE_LEVEL_PATTERN = regex.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
_DIGIT_PATTERN = regex.compile(r"\p{N}")


def _byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    chars = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + extra)
            extra += 1
    encoder = dict(zip(visible, map(chr, chars)))
    return encoder, {character: byte for byte, character in encoder.items()}


_BYTE_ENCODER, _BYTE_DECODER = _byte_alphabet()


class SmolLMTokenizer:
    """Exact local tokenizer for HuggingFaceTB/SmolLM2 Instruct files."""

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        data = json.loads((model_path / "tokenizer.json").read_text())
        config = json.loads((model_path / "tokenizer_config.json").read_text())
        expected_pre = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True,
                },
            ],
        }
        model = data["model"]
        if (
            data.get("normalizer") is not None
            or data.get("pre_tokenizer") != expected_pre
            or model.get("type") != "BPE"
            or model.get("dropout") is not None
            or model.get("unk_token") is not None
            or model.get("byte_fallback")
            or model.get("continuing_subword_prefix") is not None
            or model.get("end_of_word_suffix") is not None
            or model.get("ignore_merges")
        ):
            raise ValueError(
                "unsupported tokenizer configuration; expected SmolLM2 byte BPE"
            )

        self.vocab: dict[str, int] = model["vocab"]
        self.id_to_token = {value: key for key, value in self.vocab.items()}
        self.merge_ranks = {
            tuple(merge.split(" ", 1)): rank
            for rank, merge in enumerate(model["merges"])
        }
        added = data["added_tokens"]
        if any(
            item["lstrip"]
            or item["rstrip"]
            or item["single_word"]
            or item["normalized"]
            or not item["special"]
            for item in added
        ):
            raise ValueError("unsupported SmolLM2 added-token behavior")
        self.special_ids = {item["id"] for item in added}
        self.special_to_id = {item["content"]: item["id"] for item in added}
        self._special_pattern = regex.compile(
            "|".join(
                regex.escape(token)
                for token in sorted(self.special_to_id, key=len, reverse=True)
            )
        )
        self._chat_template = Environment(autoescape=False).from_string(
            config["chat_template"]
        )
        self.eos_token_ids = {self.special_to_id[config["eos_token"]]}
        # Cache pieces per instance so unloading a model can release its vocabulary.
        self._cached_bpe = lru_cache(maxsize=8192)(self._merge_piece)

    def _bpe(self, piece: str) -> tuple[int, ...]:
        return (
            self._cached_bpe(piece) if len(piece) <= 256 else self._merge_piece(piece)
        )

    def _merge_piece(self, piece: str) -> tuple[int, ...]:
        """Merge the lowest-ranked adjacent pair, with O(n log n) heap updates."""
        symbols = [_BYTE_ENCODER[byte] for byte in piece.encode("utf-8")]
        count = len(symbols)
        if not count:
            return ()
        before, after = list(range(-1, count - 1)), list(range(1, count + 1))
        versions = [0] * count
        pending = []

        def push(left):
            if left < 0 or left >= count:
                return
            right = after[left]
            if right >= count:
                return
            rank = self.merge_ranks.get((symbols[left], symbols[right]))
            if rank is not None:
                heapq.heappush(
                    pending, (rank, left, right, versions[left], versions[right])
                )

        for i in range(count - 1):
            push(i)
        while pending:
            _, left, right, lv, rv = heapq.heappop(pending)
            if after[left] != right or versions[left] != lv or versions[right] != rv:
                continue
            symbols[left] += symbols[right]
            versions[left] += 1
            versions[right] += 1
            after[left] = after[right]
            if after[right] < count:
                before[after[right]] = left
            after[right] = count
            push(before[left])
            push(left)
        ids, i = [], 0
        while i < count:
            ids.append(self.vocab[symbols[i]])
            i = after[i]
        return tuple(ids)

    def encode(self, text: str) -> list[int]:
        """Encode plain text, recognizing special tokens wherever they appear."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids: list[int] = []
        offset = 0
        for match in self._special_pattern.finditer(text):
            ids.extend(self._encode_plain(text[offset : match.start()]))
            ids.append(self.special_to_id[match.group()])
            offset = match.end()
        ids.extend(self._encode_plain(text[offset:]))
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        # The checkpoint's Sequence applies Unicode Digits before ByteLevel;
        # doing these in reverse changes whitespace boundaries around numerals.
        offset = 0
        spans = []
        for digit in _DIGIT_PATTERN.finditer(text):
            spans.extend((text[offset : digit.start()], digit.group()))
            offset = digit.end()
        spans.append(text[offset:])
        for span in spans:
            for match in _BYTE_LEVEL_PATTERN.finditer(span):
                ids.extend(self._bpe(match.group()))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode UTF-8 text while omitting all registered special tokens."""
        token_bytes = bytearray()
        for token_id in ids:
            if token_id in self.special_ids:
                continue
            token = self.id_to_token[token_id]
            token_bytes.extend(_BYTE_DECODER[character] for character in token)
        return token_bytes.decode("utf-8", errors="replace")

    def encode_messages(self, messages: list[dict]) -> list[int]:
        """Apply SmolLM2's chat template and append the assistant prompt."""
        if not messages:
            raise ValueError("Cannot apply chat template to an empty conversation")
        rendered = self._chat_template.render(
            messages=messages, add_generation_prompt=True
        )
        return self.encode(rendered)


class HFTokenizer:
    """Hugging Face tokenizer adapter, useful as a reference or fallback."""

    def __init__(self, model_path: str | Path):
        self.tokenizer = load_hf_tokenizer(model_path)
        eos = self.tokenizer.eos_token_id
        self.eos_token_ids = {eos} if eos is not None else set()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def encode_messages(self, messages: list[dict]) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )


def load_hf_tokenizer(model_path):
    """Preserve the checkpoint pipeline instead of reconstructing GPT2 defaults."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    path = Path(model_path)
    config = json.loads((path / "tokenizer_config.json").read_text())
    cls = (
        PreTrainedTokenizerFast
        if config.get("tokenizer_class") in {"GPT2Tokenizer", "GPT2TokenizerFast"}
        else AutoTokenizer
    )
    return cls.from_pretrained(path, local_files_only=True, trust_remote_code=False)
