import torch
import torch.nn as nn
import numpy as np
from fairseq.models.speech_to_text.xstnet import XSTNetEncoder
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.models.fairseq_encoder import EncoderOut
import logging

logger = logging.getLogger(__name__)

class MoyuAttnLayer(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.input_size = input_size
        self.expand_size = input_size * 2
        self.expand_adapter = self._build_ffn(self.input_size, self.expand_size)
        self.forget_gate = self._build_ffn(self.expand_size, 1)
        self.update_gate = self._build_ffn(self.expand_size, 1)
        self.multi_head_attn = MultiheadAttention(
            self.expand_size,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
        self.shrink_adapter = self._build_ffn(self.expand_size, self.input_size)
    
    def _build_ffn(self, input_size, expand_size):
        ffn_layer = nn.Linear(input_size, expand_size)
        sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(ffn_layer.weight)
        return nn.Sequential(ffn_layer, sigmoid)
    
    def forward(self, x):
        x = self.expand_adapter(x)
        forget_value = self.forget_gate(x)
        update_value = self.update_gate(x)
        attn_x, _ = self.multi_head_attn(x, x, x)
        gated_value = attn_x + update_value * attn_x + (1 - forget_value) * attn_x
        x = self.shrink_adapter(gated_value)
        return x

class MoyuNetEncoder(XSTNetEncoder):
    def __init__(self, args, dict, embed_tokens):
        super().__init__(args, dict, embed_tokens)
        self.using_moyulayer = args.using_attn
        if self.using_moyulayer:
            self.Moyu_layer = MoyuAttnLayer(args, self.args.encoder_embed_dim)
    
    def forward(self, src_tokens, src_lengths, is_text_input=False, **kwargs):
        """
        src_tokens: b x seq, float tensor if it is audio input, LongTensor if it is text input
        src_lengths: b-dim LongTensor
        """
        short_audio_len = None
        if self.is_text_input:
            is_text_input = True
        if is_text_input:
            x, encoder_padding_mask = self.embedding_text(src_tokens, src_lengths)
        else:
            x, encoder_padding_mask, short_audio_len = self.embedding_audio(src_tokens, src_lengths,
                                                                            return_short_audio_len=True)
            if self.using_moyulayer:
                x = self.Moyu_layer(x)

        encoder_embedding = x
        # 3. Transformer-layers
        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=encoder_embedding,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
            output_encoder_lengths=short_audio_len,
        )