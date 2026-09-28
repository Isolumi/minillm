import threading
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Literal["stop", "length"]


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
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> GenerationResult:
        with self._lock, torch.inference_mode():
            return self._generate_locked(messages, max_new_tokens)

    def _generate_locked(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> GenerationResult:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        prompt_length = inputs.input_ids.shape[-1]

        if prompt_length + max_new_tokens > self.context_limit:
            raise ContextLimitExceeded("convo too long")

        result = self.model.generate(
            **inputs,
            past_key_values=DynamicCache(config=self.model.config),
            use_cache=True,
            return_dict_in_generate=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

        new_tokens = result.sequences[0, prompt_length:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        eos_token_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        finish_reason = (
            "stop"
            if len(new_tokens) and int(new_tokens[-1]) in (eos_token_ids or [])
            else "length"
        )

        return GenerationResult(
            text=answer,
            prompt_tokens=prompt_length,
            completion_tokens=len(new_tokens),
            finish_reason=finish_reason,
        )
