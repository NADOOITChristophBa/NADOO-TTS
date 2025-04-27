import torch
from dia.config import DiaConfig
from dia.state import DecoderInferenceState, KVCache
from dia.onnx_wrapper import DecoderONNXWrapper
from dia.onnx_decoder import DecoderForONNX

# 1. Config und Decoder laden
def main():
    config = DiaConfig.load("config.json")
    compute_dtype = torch.float32
    device = torch.device("cpu")

    decoder = DecoderForONNX(config, compute_dtype).to(device)
    decoder.eval()
    decoder_onnx = DecoderONNXWrapper(decoder)

    # 2. Dummy-Input erzeugen
    batch_size = 1
    seq_len = 1  # Für ONNX-Export Schrittweite 1
    channels = config.data.channels
    vocab_size = config.model.tgt_vocab_size

    dummy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len, channels), dtype=torch.long, device=device)

    # 3. Dummy-State erzeugen
    num_layers = config.model.decoder.n_layer
    kv_heads = config.model.decoder.kv_heads
    gqa_head_dim = config.model.decoder.gqa_head_dim
    cross_query_heads = config.model.decoder.cross_query_heads
    cross_head_dim = config.model.decoder.cross_head_dim
    max_len = seq_len

    # Self-Attention KVCache (kv_heads)
    self_attn_kvcache = [KVCache(kv_heads, max_len, gqa_head_dim, compute_dtype, device) for _ in range(num_layers)]
    # Cross-Attention KVCache (cross_query_heads)
    cross_attn_kvcache = [KVCache(cross_query_heads, max_len, cross_head_dim, compute_dtype, device) for _ in range(num_layers)]

    enc_out = torch.zeros((batch_size, seq_len, config.model.encoder.n_embd), dtype=compute_dtype, device=device)
    enc_positions = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    dec_positions = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    dec_cross_attn_mask = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.bool, device=device)

    # Zerlege alle KVCache-Objekte in k/v-Tensoren (erst self, dann cross, pro Layer)
    kv_tensors = []
    for i in range(num_layers):
        kv_tensors.append(self_attn_kvcache[i].k)
        kv_tensors.append(self_attn_kvcache[i].v)
        kv_tensors.append(cross_attn_kvcache[i].k)
        kv_tensors.append(cross_attn_kvcache[i].v)

    # 4. ONNX-Export
    try:
        dynamic_axes = {
            "tgt_ids_BxTxC": {0: "batch", 1: "seq"},
            "enc_out": {0: "batch", 1: "seq_enc"},
            "enc_positions": {0: "batch", 1: "seq_enc"},
            "dec_positions": {0: "batch", 1: "seq"},
            "dec_cross_attn_mask": {0: "batch", 2: "seq", 3: "seq_enc"},
        }
        for i in range(len(kv_tensors)):
            # Annahme: alle kv_tensors haben [batch, heads, seq, dim] oder [batch, heads, seq_enc, dim]
            dynamic_axes[f"kv_{i}"] = {0: "batch", 2: f"seq_kv_{i}"}
        torch.onnx.export(
            decoder_onnx,
            (dummy_tokens, enc_out, enc_positions, dec_positions, dec_cross_attn_mask, *kv_tensors),
            "dia_decoder.onnx",
            input_names=["tgt_ids_BxTxC", "enc_out", "enc_positions", "dec_positions", "dec_cross_attn_mask"] + [f"kv_{i}" for i in range(len(kv_tensors))],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes=dynamic_axes,
        )
        print("ONNX-Export abgeschlossen!")
    except Exception as e:
        print(f"ONNX-Export fehlgeschlagen: {e}")

if __name__ == "__main__":
    main()
