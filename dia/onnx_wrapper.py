import torch
from dia.layers import Decoder
from dia.state import KVCache, DecoderInferenceState

class DecoderONNXWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        tgt_ids_BxTxC,
        enc_out,
        enc_positions,
        dec_positions,
        dec_cross_attn_mask,
        *kv_caches  # alle KVCache k/v für alle Layer, als flache Liste
    ):
        num_layers = len(kv_caches) // 4  # für jede Schicht: self_k, self_v, cross_k, cross_v
        self_attn_cache = []
        cross_attn_cache = []
        for i in range(num_layers):
            self_attn_cache.append(KVCache.from_kv(kv_caches[4*i], kv_caches[4*i+1]))
            cross_attn_cache.append(KVCache.from_kv(kv_caches[4*i+2], kv_caches[4*i+3]))
        state = DecoderInferenceState(
            device=tgt_ids_BxTxC.device,
            dtype=tgt_ids_BxTxC.dtype,
            enc_out=enc_out,
            enc_positions=enc_positions,
            dec_positions=dec_positions,
            dec_cross_attn_mask=dec_cross_attn_mask,
            self_attn_cache=self_attn_cache,
            cross_attn_cache=cross_attn_cache,
        )
        return self.decoder(tgt_ids_BxTxC, state)
