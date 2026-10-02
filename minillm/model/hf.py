"""HF oracle and architecture-specific text/image runners, using our decode loop."""

import base64
import binascii
import io
from dataclasses import dataclass, field

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from minillm.tokenization.smollm import load_hf_tokenizer


MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096


def image_from_data_uri(uri: str) -> Image.Image:
    header, sep, encoded = uri.partition(",")
    if not sep or header.lower() not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
        raise ValueError("Images must be base64 PNG, JPEG, or WebP data URIs")
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 4):
        raise ValueError("Image exceeds the 4 MiB limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 4 MiB limit")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel limit")
            return image.convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid image data") from exc


def text_messages(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            if any(part.get("type") not in {"input_text", "output_text", "text"} for part in content):
                raise ValueError("This model accepts text only; select the multimodal model for images")
            content = "".join(part["text"] for part in content)
        result.append({"role": message["role"], "content": content})
    return result


@dataclass
class HFState:
    length: int = 0
    cache: object = None
    initial_inputs: dict = field(default_factory=dict)
    released: bool = False


class HFRunner:
    def __init__(self, model_path: str, device="cuda", dtype=torch.float16,
                 max_context=8192, multimodal=False, attention_backend="sdpa"):
        self.device = torch.device(device)
        self.multimodal = multimodal
        cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
        # Gemma checkpoints use BF16; use their native dtype to avoid FP16 overflow.
        if multimodal and self.device.type == "cuda" and dtype == torch.float16:
            dtype = torch.bfloat16
        self.model = cls.from_pretrained(model_path, local_files_only=True,
                                        dtype=dtype, attn_implementation=attention_backend).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True) if multimodal else None
        self.tokenizer = self.processor.tokenizer if self.processor else load_hf_tokenizer(model_path)
        cfg = self.model.config.get_text_config()
        self.context_limit = min(max_context, cfg.max_position_embeddings)
        eos = self.model.generation_config.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        self._states: list[HFState] = []

    def encode_messages(self, messages: list[dict]) -> tuple[list[int], dict]:
        if not self.multimodal:
            return self.tokenizer.apply_chat_template(text_messages(messages), tokenize=True, add_generation_prompt=True, return_dict=False), {}
        formatted = []
        images = 0
        for message in messages:
            content = message["content"]
            parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
            output = []
            for part in parts:
                if part["type"] in {"input_text", "output_text", "text"}:
                    output.append({"type": "text", "text": part["text"]})
                elif part["type"] == "input_image":
                    images += 1
                    if images > 1:
                        raise ValueError("At most one image is supported in a conversation")
                    output.append({"type": "image", "image": image_from_data_uri(part["image_url"])})
                else:
                    raise ValueError(f"Unsupported content type: {part['type']}")
            formatted.append({"role": message["role"], "content": output})
        inputs = self.processor.apply_chat_template(formatted, tokenize=True, return_dict=True,
                                                   return_tensors="pt", add_generation_prompt=True,
                                                   enable_thinking=False)
        ids = inputs.pop("input_ids")[0].tolist()
        inputs.pop("attention_mask", None)
        return ids, dict(inputs)

    def decode_text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def create_state(self):
        state = HFState()
        self._states.append(state)
        return state

    @torch.inference_mode()
    def prefill(self, token_ids: list[int], state: HFState) -> torch.Tensor:
        if state.released or not token_ids:
            raise ValueError("Cannot append to a released state or append zero tokens")
        end = state.length + len(token_ids)
        if end > self.context_limit:
            raise ValueError("Context limit exceeded")
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        extras = state.initial_inputs if state.length == 0 else {}
        extras = {k: v.to(self.device, dtype=self.model.dtype) if torch.is_tensor(v) and v.is_floating_point()
                  else v.to(self.device) if torch.is_tensor(v) else v for k, v in extras.items()}
        model_inputs = self.model.prepare_inputs_for_generation(
            ids, past_key_values=state.cache,
            attention_mask=torch.ones((1, end), dtype=torch.long, device=self.device),
            position_ids=torch.arange(state.length, end, device=self.device).unsqueeze(0),
            is_first_iteration=state.length == 0, use_cache=True, **extras,
        )
        result = self.model(**model_inputs, logits_to_keep=1, return_dict=True)
        state.cache = result.past_key_values
        state.length = end
        state.initial_inputs = {}
        return result.logits[0, -1]

    def decode(self, token_ids: list[int], states: list[HFState]) -> torch.Tensor:
        # HF architectures have independent DynamicCaches; custom SmolLM2 batches.
        return torch.stack([self.prefill([token], state) for token, state in zip(token_ids, states, strict=True)])

    def release(self, state: HFState):
        if state.released:
            return
        state.cache = None
        state.initial_inputs = {}
        state.released = True
        self._states = [s for s in self._states if s is not state]

    def stats(self):
        cache_bytes = 0
        for state in self._states:
            for layer in getattr(state.cache, "layers", []):
                for key in ("keys", "values"):
                    tensor = getattr(layer, key, None)
                    if torch.is_tensor(tensor):
                        cache_bytes += tensor.numel() * tensor.element_size()
        return {"backend": "hf", "active_states": len(self._states), "cache_bytes": cache_bytes,
                "multimodal": self.multimodal}
