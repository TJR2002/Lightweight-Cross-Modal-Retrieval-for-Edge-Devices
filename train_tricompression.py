"""
Improved Tri-Compression Training Script
Fixes accuracy issues with better hyperparameters and training strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

# Import from main module
from cmrs_tricompression import (
    StudentEncoder, config, DistillationDataset,
    TriCompressionTrainer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedDistillationTrainer:
    """Improved trainer with better loss functions and training strategy"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load CLIP teacher
        logger.info("Loading CLIP teacher model...")
        self.teacher_model, self.preprocess = clip.load(config.CLIP_MODEL, device=self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Projection layer for teacher features
        self.teacher_projection = nn.Linear(512, config.EMBEDDING_DIM).to(self.device)
        
        self.tokenizer = clip.tokenize
    
    def improved_distillation_loss(self, student_img, student_txt, teacher_img, teacher_txt):
        """
        Improved loss combining:
        1. Feature matching (MSE)
        2. Cosine similarity preservation
        3. Contrastive alignment
        """
        # Project teacher features
        teacher_img_proj = self.teacher_projection(teacher_img)
        teacher_txt_proj = self.teacher_projection(teacher_txt)
        teacher_img_proj = F.normalize(teacher_img_proj, dim=-1)
        teacher_txt_proj = F.normalize(teacher_txt_proj, dim=-1)
        
        # 1. Feature MSE loss
        mse_img = F.mse_loss(student_img, teacher_img_proj)
        mse_txt = F.mse_loss(student_txt, teacher_txt_proj)
        
        # 2. Cosine similarity preservation
        # Preserve relative similarities between teacher and student
        teacher_sim = torch.matmul(teacher_img_proj, teacher_txt_proj.T)
        student_sim = torch.matmul(student_img, student_txt.T)
        sim_loss = F.mse_loss(student_sim, teacher_sim)
        
        # 3. Contrastive loss
        batch_size = student_img.shape[0]
        labels = torch.arange(batch_size).to(self.device)
        
        # Scale by learnable temperature
        logit_scale = torch.tensor(np.log(1 / 0.07)).to(self.device)
        
        logits_per_image = logit_scale * student_sim
        logits_per_text = logits_per_image.T
        
        contrast_loss_img = F.cross_entropy(logits_per_image, labels)
        contrast_loss_txt = F.cross_entropy(logits_per_text, labels)
        contrast_loss = (contrast_loss_img + contrast_loss_txt) / 2
        
        # Combined loss with weights
        total_loss = (
            0.3 * (mse_img + mse_txt) / 2 +  # Feature matching
            0.3 * sim_loss +                   # Similarity preservation
            0.4 * contrast_loss                # Contrastive alignment
        )
        
        return total_loss, mse_img, mse_txt, sim_loss, contrast_loss
    
    def train_improved(self, train_dataset, val_dataset, epochs=20, batch_size=64, lr=2e-4):
        """Train with improved strategy"""
        
        # Initialize student
        student_model = StudentEncoder(
            output_dim=config.EMBEDDING_DIM,
            hidden_dim=config.STUDENT_HIDDEN_DIM,
            num_layers=config.STUDENT_NUM_LAYERS
        ).to(self.device)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            list(student_model.parameters()) + list(self.teacher_projection.parameters()),
            lr=lr,
            weight_decay=0.01
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,
            drop_last=True
        )
        
        logger.info(f"Training for {epochs} epochs...")
        logger.info(f"Batch size: {batch_size}, Learning rate: {lr}")
        
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            student_model.train()
            self.teacher_projection.train()
            
            total_loss = 0
            total_mse_img = 0
            total_mse_txt = 0
            total_sim = 0
            total_contrast = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, captions in pbar:
                images = images.to(self.device)
                texts = self.tokenizer(captions, truncate=True).to(self.device)
                
                # Teacher forward
                with torch.no_grad():
                    teacher_img_features = self.teacher_model.encode_image(images)
                    teacher_txt_features = self.teacher_model.encode_text(texts)
                    teacher_img_features = F.normalize(teacher_img_features.float(), dim=-1)
                    teacher_txt_features = F.normalize(teacher_txt_features.float(), dim=-1)
                
                # Student forward
                student_img_features, student_txt_features = student_model(images, texts)
                
                # Improved loss
                loss, mse_img, mse_txt, sim_loss, contrast_loss = self.improved_distillation_loss(
                    student_img_features, student_txt_features,
                    teacher_img_features, teacher_txt_features
                )
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.teacher_projection.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_mse_img += mse_img.item()
                total_mse_txt += mse_txt.item()
                total_sim += sim_loss.item()
                total_contrast += contrast_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse_img': f'{mse_img.item():.4f}',
                    'contrast': f'{contrast_loss.item():.4f}'
                })
            
            scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                f"MSE_img={total_mse_img/len(train_loader):.4f}, "
                f"MSE_txt={total_mse_txt/len(train_loader):.4f}, "
                f"Sim={total_sim/len(train_loader):.4f}, "
                f"Contrast={total_contrast/len(train_loader):.4f}"
            )
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'projection_state_dict': self.teacher_projection.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, config.STUDENT_MODEL_PATH)
                
                logger.info(f"Saved best model (loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training complete!")
        return student_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to training images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=10000, help='Max training samples')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("IMPROVED TRI-COMPRESSION TRAINING")
    logger.info("="*80)
    
    # Prepare data
    logger.info("Preparing training data...")
    data_dir = Path(args.data_dir)
    
    image_paths = []
    for ext in config.SUPPORTED_IMAGE_FORMATS:
        image_paths.extend(list(data_dir.rglob(f"*{ext}")))
    
    image_paths = [str(p) for p in image_paths[:args.max_samples]]
    logger.info(f"Found {len(image_paths)} images")
    
    # Generate captions
    logger.info("Generating captions...")
    captions = []
    for img_path in tqdm(image_paths, desc="Captioning"):
        # Simple caption based on filename
        filename = Path(img_path).stem.replace('_', ' ').replace('-', ' ')
        captions.append(f"an image of {filename}")
    
    # Split train/val
    split_idx = int(len(image_paths) * 0.9)
    train_paths = image_paths[:split_idx]
    train_captions = captions[:split_idx]
    val_paths = image_paths[split_idx:]
    val_captions = captions[split_idx:]
    
    logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}")
    
    train_dataset = DistillationDataset(train_paths, train_captions)
    val_dataset = DistillationDataset(val_paths, val_captions)
    
    # Train
    trainer = ImprovedDistillationTrainer()
    student_model = trainer.train_improved(
        train_dataset, val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    logger.info("="*80)
    logger.info("Stage 1 (Improved Distillation) Complete!")
    logger.info(f"Model saved to {config.STUDENT_MODEL_PATH}")
    logger.info("="*80)
    
    # Continue with pruning and quantization using original trainer
    logger.info("\nContinuing with Stage 2 (Pruning) and Stage 3 (Quantization)...")
    original_trainer = TriCompressionTrainer()
    
    # Pruning
    pruned_model = original_trainer.prune_model(student_model, train_dataset)
    logger.info(f"Stage 2 complete! Saved to {config.PRUNED_MODEL_PATH}")
    
    # Quantization
    original_trainer.quantize_model(pruned_model, train_dataset)
    logger.info(f"Stage 3 complete! Saved to {config.QUANTIZED_MODEL_PATH}")
    
    logger.info("="*80)
    logger.info("ALL 3 STAGES COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()