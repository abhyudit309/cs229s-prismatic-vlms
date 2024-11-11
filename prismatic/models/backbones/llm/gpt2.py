"""
gpt2.py

Class definition for all LLMs derived from GPT2LMHeadModel.
"""

from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder, GPT2PromptBuilder

# Registry =>> Support GPT2 Models (from HF Transformers)
# fmt: off
GPT2_MODELS = {
    "gpt2-small": {
        "llm_family": "gpt2", "llm_cls": GPT2LMHeadModel, "hf_hub_path": "openai-community/gpt2"
    },
    
    "gpt2-medium": {
        "llm_family": "gpt2", "llm_cls": GPT2LMHeadModel, "hf_hub_path": "openai-community/gpt2-medium"
    },

    "gpt2-large": {
        "llm_family": "gpt2", "llm_cls": GPT2LMHeadModel, "hf_hub_path": "openai-community/gpt2-large"
    },

    "gpt2-xl": {
        "llm_family": "gpt2", "llm_cls": GPT2LMHeadModel, "hf_hub_path": "openai-community/gpt2-xl"
    },
}
# fmt: on


class GPT2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 1024,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **GPT2_MODELS[llm_backbone_id],
        )

        # [Special Case] GPT2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.startswith("gpt2"):
            return GPT2PromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GPT2Block

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16