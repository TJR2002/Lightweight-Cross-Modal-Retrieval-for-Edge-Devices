"""
Fixed Stage 3: Quantize only projection layers (avoids transformer issues)
"""
import torch
import torch.nn as nn
import logging
import os
from cmrs_tricompression import StudentEncoder, Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = Config()

if __name__ == '__main__':
    print("Loading pruned model from Stage 2...")
    
    model = StudentEncoder(
        output_dim=config.EMBEDDING_DIM,
        hidden_dim=config.STUDENT_HIDDEN_DIM,
        num_layers=config.STUDENT_NUM_LAYERS
    )
    
    checkpoint = torch.load(config.PRUNED_MODEL_PATH, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded from {checkpoint.get('timestamp', 'unknown')}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("\nRunning Stage 3: Selective Quantization")
    print("Quantizing projection layers only (keeps transformers stable)\n")
    
    # Quantize only projections
    model.vision_projection = torch.quantization.quantize_dynamic(
        model.vision_projection, {nn.Linear}, dtype=torch.qint8
    )
    print("  ✓ Quantized vision_projection")
    
    model.text_projection = torch.quantization.quantize_dynamic(
        model.text_projection, {nn.Linear}, dtype=torch.qint8
    )
    print("  ✓ Quantized text_projection")
    
    print(f"\n✓ Saving to {config.QUANTIZED_MODEL_PATH}")
    torch.save(model, config.QUANTIZED_MODEL_PATH)
    
    size_mb = os.path.getsize(config.QUANTIZED_MODEL_PATH) / (1024*1024)
    print(f"✓ Saved! Size: {size_mb:.2f} MB\n")
    print("Quantization complete - model is ready to use!")