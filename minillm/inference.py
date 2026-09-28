import threading
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)

from minillm.conversations import Conversation


@dataclass
class GenerationResult:
    text: str
    reused_tokens: int


class ContextLimitExceeded(Exception):
    pass


class InferenceEngine:
    def __init__(self, model_path: str):
        if not torch.cuda.is_available():
            raise RuntimeError("where cuda :(")

        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

        self.context_limit = self.model.config.max_position_embeddings

        self._lock = threading.Lock()

    def generate(
        self,
        conversation: Conversation,
        prompt: str,
        max_new_tokens: int,
    ) -> GenerationResult:
        with self._lock, torch.inference_mode():
            return self._generate_locked(conversation, prompt, max_new_tokens)

    def _generate_locked(
        self,
        conversation: Conversation,
        prompt: str,
        max_new_tokens: int,
    ) -> GenerationResult:
        messages = conversation.messages + [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        prompt_ids = inputs.input_ids[0].tolist()
        prompt_length = len(prompt_ids)

        if prompt_length + max_new_tokens > self.context_limit:
            raise ContextLimitExceeded("convo too long")

        reused_tokens = 0

        if conversation.cache is not None:
            cached_length = conversation.cache.get_seq_length()

            prefix_matches = (
                cached_length == len(conversation.cached_ids)
                and cached_length < prompt_length
                and prompt_ids[:cached_length] == conversation.cached_ids
            )

            if prefix_matches:
                reused_tokens = cached_length
            else:
                conversation.cache = None
                conversation.cached_ids = []

        # not else branch because prev if can delete cache
        if conversation.cache is None:
            conversation.cache = DynamicCache(config=self.model.config)

        try:
            result = self.model.generate(
                **inputs,
                past_key_values=conversation.cache,
                use_cache=True,
                return_dict_in_generate=True,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

            new_tokens = result.sequences[0, prompt_length:]
            answer = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )

            conversation.cache = result.past_key_values
            cached_length = conversation.cache.get_seq_length()
            conversation.cached_ids = (
                result.sequences[0, :cached_length].tolist()
            )

            conversation.messages = messages + [
                {"role": "assistant", "content": answer}
            ]

            return GenerationResult(
                text=answer,
                reused_tokens=reused_tokens,
            )

        except Exception:
            conversation.cache = None
            conversation.cached_ids = []
            raise