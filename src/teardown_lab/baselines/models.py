# ABOUTME: Backends for the zero-shot baselines: NVIDIA Cosmos 3 Edge (world model) and
# ABOUTME: Ai2 MolmoAct 2 (VLA), each exposed as a frame+prompt -> text responder.

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

COSMOS_REPO = "nvidia/Cosmos3-Edge"
MOLMOACT_REPO = "allenai/MolmoAct2"

# Short replies: we want one small JSON object, and every extra token is latency on a
# control loop that already runs far slower than the game.
MAX_NEW_TOKENS = 96


@dataclass
class ResponderStats:
    calls: int = 0
    latencies_ms: list = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.calls += 1
        self.latencies_ms.append(ms)

    @property
    def median_ms(self) -> float:
        return float(np.median(self.latencies_ms)) if self.latencies_ms else 0.0


def _to_pil(frame: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.asarray(frame, dtype=np.uint8))


class HFChatResponder:
    """image + prompt -> text, for HF models exposing a chat template.

    Both baselines are loaded in bfloat16: Cosmos 3 Edge is bf16-only (fp16/fp8/fp4 are
    explicitly unsupported), and MolmoAct 2 fits a 24 GB card comfortably in bf16.
    """

    def __init__(
        self,
        repo: str,
        device: str = "cuda",
        max_new_tokens: int = MAX_NEW_TOKENS,
        auto_class: str = "AutoModelForImageTextToText",
        load_in_4bit: bool = False,
    ):
        import torch
        import transformers
        from transformers import AutoProcessor

        self.repo = repo
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.stats = ResponderStats()
        self._torch = torch

        self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        # Auto class differs per checkpoint: MolmoAct 2 declares AutoModelForImageTextToText
        # in its auto_map, while Cosmos 3 Edge needs its own architecture class.
        loader = getattr(transformers, auto_class)
        kwargs = dict(device_map=device, trust_remote_code=True)
        if load_in_4bit:
            # This GPU is shared with another service of Louis's holding ~13 GB, so a 5B
            # model in bf16 does not fit. 4-bit keeps the baseline runnable without
            # disturbing that process; the quantisation is reported alongside results.
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["dtype"] = torch.bfloat16
        self.model = loader.from_pretrained(repo, **kwargs).eval()

    def __call__(self, frame: np.ndarray, prompt: str) -> str:
        started = time.perf_counter()
        image = _to_pil(frame)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = None
        try:
            # Preferred path: let the processor tokenise image+text together. Cosmos 3
            # Edge only accepts this form; MolmoAct 2 accepts either.
            messages[0]["content"][0]["image"] = image
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        except Exception:
            inputs = None

        if inputs is None:
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = prompt
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(
                self.device
            )
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[0][inputs["input_ids"].shape[-1] :]
        reply = self.processor.decode(trimmed, skip_special_tokens=True)
        self.stats.record((time.perf_counter() - started) * 1000)
        return reply


# Cosmos 3 Edge thinks out loud before answering - its autoregressive reasoning tower
# emits several hundred tokens of deliberation, so a 96-token cap truncates it before any
# JSON appears. 640 is the smallest cap measured to reliably reach a parseable action.
COSMOS_MAX_NEW_TOKENS = 640


def load_cosmos_edge(device: str = "cuda") -> HFChatResponder:
    """NVIDIA Cosmos 3 Edge, the world-model side of the comparison.

    Needs transformers from git: `cosmos3_edge` is absent from release 5.14.1 and the
    checkpoint ships neither an auto_map nor modeling code.
    """
    return HFChatResponder(
        COSMOS_REPO,
        device=device,
        max_new_tokens=COSMOS_MAX_NEW_TOKENS,
        auto_class="Cosmos3EdgeForConditionalGeneration",
    )


def load_molmoact2(device: str = "cuda", load_in_4bit: bool = False) -> HFChatResponder:
    """Ai2 MolmoAct 2, the VLA side of the comparison.

    Driven as a VLM: its action head needs a robot joint-state vector and one of a closed
    set of normalisation tags, neither of which exists for a game.

    bf16 by default: it fits a 24 GB card at 10.9 GB, and the 4-bit path fed the vision
    tower uint8 tensors ("LayerNormKernelImpl not implemented for Byte"). Only quantise
    if something else is occupying the GPU.
    """
    return HFChatResponder(MOLMOACT_REPO, device=device, load_in_4bit=load_in_4bit)


LOADERS = {"cosmos_edge": load_cosmos_edge, "molmoact2": load_molmoact2}
