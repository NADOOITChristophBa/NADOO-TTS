import torch
import torch.nn as nn
from .config import DiaConfig
from .state import DecoderInferenceState, KVCache
from .layers import DenseGeneral, MlpBlock, Attention

class DecoderLayerForONNX(nn.Module):
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        model_config = config.model
        dec_config = config.model.decoder
        enc_config = config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd
        self.compute_dtype = compute_dtype

        # Norms (LayerNorm statt RMSNorm)
        self.pre_sa_norm = nn.LayerNorm(dec_embed_dim, eps=model_config.normalization_layer_epsilon)
        self.pre_ca_norm = nn.LayerNorm(dec_embed_dim, eps=model_config.normalization_layer_epsilon)
        self.pre_mlp_norm = nn.LayerNorm(dec_embed_dim, eps=model_config.normalization_layer_epsilon)

        # Self-Attention (GQA)
        self.self_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.gqa_query_heads,
            num_kv_heads=dec_config.kv_heads,
            head_dim=dec_config.gqa_head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=False,
            out_embed_dim=dec_embed_dim,
        )
        # Cross-Attention (MHA)
        self.cross_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,
            num_query_heads=dec_config.cross_query_heads,
            num_kv_heads=dec_config.cross_query_heads,
            head_dim=dec_config.cross_head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=True,
            out_embed_dim=dec_embed_dim,
        )
        # MLP
        self.mlp = MlpBlock(
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
            compute_dtype=compute_dtype,
        )

    def forward(self, x, state, self_attn_cache, cross_attn_cache, prefill=True):
        residual = x
        x_norm = self.pre_sa_norm(x)
        sa_out = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=state.dec_positions,
            kv_positions=state.dec_positions,
            attn_mask=None,
            cache=self_attn_cache,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.cross_attention(
            Xq=x_norm,
            Xkv=state.enc_out,
            q_positions=state.dec_positions,
            kv_positions=state.enc_positions,
            attn_mask=state.dec_cross_attn_mask,
            cache=cross_attn_cache,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x

class DecoderForONNX(nn.Module):
    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        model_config = config.model
        dec_config = config.model.decoder
        data_config = config.data
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer

        self.embeddings = nn.ModuleList([
            nn.Embedding(model_config.tgt_vocab_size, dec_config.n_embd) for _ in range(self.num_channels)
        ])
        self.layers = nn.ModuleList([
            DecoderLayerForONNX(config=config, compute_dtype=compute_dtype) for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(dec_config.n_embd, eps=model_config.normalization_layer_epsilon)
        self.logits_dense = DenseGeneral(
            in_shapes=(dec_config.n_embd,),
            out_features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )

    def forward(self, tgt_ids_BxTxC, state: DecoderInferenceState):
        _, _, num_channels_in = tgt_ids_BxTxC.shape
        assert num_channels_in == self.num_channels, "Input channels mismatch"
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i]
            channel_embed = self.embeddings[i](channel_tokens)
            x = channel_embed if x is None else x + channel_embed
        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(x, state, self_attn_cache=self_cache, cross_attn_cache=cross_cache, prefill=True)
        x = self.norm(x)
        logits_BxTxCxV = self.logits_dense(x)
        return logits_BxTxCxV.to(torch.float32)
