"""
Training script for Egyptian Arabic TTS model.

This script handles the complete training pipeline including data loading,
model training, validation, and checkpoint management.
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import our model and preprocessing modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.egyptian_tts import EgyptianArabicTTS, create_model_config
from preprocessing.text_processor import EgyptianArabicProcessor
from preprocessing.audio_processor import AudioProcessor


class EgyptianTTSDataset(Dataset):
    """
    Dataset class for Egyptian Arabic TTS training data.
    Handles text-audio pairs with proper preprocessing and augmentation.
    """
    
    def __init__(self, data_dir: str, config: Dict, split: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing training data
            config: Training configuration
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.config = config
        self.split = split
        
        # Initialize processors
        self.text_processor = EgyptianArabicProcessor()
        self.audio_processor = AudioProcessor(
            target_sr=config['sample_rate'],
            target_bit_depth=config['bit_depth']
        )
        
        # Load data manifest
        self.data_manifest = self.load_manifest()
        
        # Setup augmentation for training
        self.use_augmentation = (split == 'train') and config.get('use_augmentation', True)
        
    def load_manifest(self) -> List[Dict]:
        """Load data manifest file."""
        manifest_path = os.path.join(self.data_dir, f'{self.split}_manifest.json')
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        logging.info(f"Loaded {len(manifest)} samples for {self.split} split")
        return manifest
    
    def __len__(self) -> int:
        return len(self.data_manifest)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed text and audio data
        """
        sample = self.data_manifest[idx]
        
        # Load and process text
        text = sample['text']
        processed_text = self.text_processor.process_for_tts(text)
        
        # Load and process audio
        audio_path = os.path.join(self.data_dir, sample['audio_path'])
        audio, sr = self.audio_processor.load_audio(audio_path)
        
        # Enhance audio quality if needed
        if not self.audio_processor.assess_quality(audio, sr)['meets_vo_standards']:
            audio, _ = self.audio_processor.enhance_audio(audio, sr)
        
        # Resample to target sample rate
        if sr != self.config['sample_rate']:
            audio = self.audio_processor.resample_audio(audio, sr, self.config['sample_rate'])
        
        # Extract mel-spectrogram
        mel_spectrogram = self.extract_mel_spectrogram(audio)
        
        # Apply augmentation if training
        if self.use_augmentation:
            mel_spectrogram = self.apply_augmentation(mel_spectrogram)
        
        # Convert to tensors
        text_tokens = self.text_to_tokens(processed_text['text'])
        phonemes = self.phonemes_to_tokens(processed_text['phonemes'])
        
        return {
            'text_tokens': torch.tensor(text_tokens, dtype=torch.long),
            'phonemes': torch.tensor(phonemes, dtype=torch.long),
            'mel_spectrogram': torch.tensor(mel_spectrogram, dtype=torch.float32),
            'audio': torch.tensor(audio, dtype=torch.float32),
            'speaker_id': sample.get('speaker_id', 'default'),
            'metadata': processed_text['metadata']
        }
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram from audio."""
        import librosa
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config['sample_rate'],
            n_mels=self.config['mel_dim'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length'],
            fmin=self.config['fmin'],
            fmax=self.config['fmax']
        )
        
        # Convert to log scale
        log_mel = np.log(mel_spec + 1e-8)
        
        return log_mel
    
    def apply_augmentation(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Apply data augmentation to mel-spectrogram."""
        # Time masking
        if np.random.random() < 0.3:
            mel_spectrogram = self.time_mask(mel_spectrogram)
        
        # Frequency masking
        if np.random.random() < 0.3:
            mel_spectrogram = self.freq_mask(mel_spectrogram)
        
        # Noise injection
        if np.random.random() < 0.2:
            noise_level = np.random.uniform(0.01, 0.05)
            mel_spectrogram += np.random.normal(0, noise_level, mel_spectrogram.shape)
        
        return mel_spectrogram
    
    def time_mask(self, mel_spec: np.ndarray, max_mask_size: int = 10) -> np.ndarray:
        """Apply time masking augmentation."""
        mel_spec = mel_spec.copy()
        mask_size = np.random.randint(1, max_mask_size)
        start_pos = np.random.randint(0, max(1, mel_spec.shape[1] - mask_size))
        mel_spec[:, start_pos:start_pos + mask_size] = 0
        return mel_spec
    
    def freq_mask(self, mel_spec: np.ndarray, max_mask_size: int = 5) -> np.ndarray:
        """Apply frequency masking augmentation."""
        mel_spec = mel_spec.copy()
        mask_size = np.random.randint(1, max_mask_size)
        start_pos = np.random.randint(0, max(1, mel_spec.shape[0] - mask_size))
        mel_spec[start_pos:start_pos + mask_size, :] = 0
        return mel_spec
    
    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to token indices."""
        # Simplified tokenization - in practice, use proper tokenizer
        return [ord(c) % 1000 for c in text[:50]]  # Limit length and vocab
    
    def phonemes_to_tokens(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to token indices."""
        # Simplified phoneme tokenization
        phoneme_vocab = {
            'b': 1, 't': 2, 'g': 3, 'ħ': 4, 'x': 5, 'd': 6, 'r': 7, 'z': 8,
            's': 9, 'ʃ': 10, 'sˤ': 11, 'dˤ': 12, 'tˤ': 13, 'ʕ': 14, 'ɣ': 15,
            'f': 16, 'ʔ': 17, 'k': 18, 'l': 19, 'm': 20, 'n': 21, 'h': 22,
            'w': 23, 'j': 24, 'a': 25, 'i': 26, 'u': 27, 'aː': 28, 'iː': 29,
            'uː': 30, '_': 31, '|': 32
        }
        
        return [phoneme_vocab.get(p, 0) for p in phonemes[:50]]  # Limit length


class TTSTrainer:
    """
    Trainer class for Egyptian Arabic TTS model.
    Handles training loop, validation, and checkpoint management.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.model = EgyptianArabicTTS(config['model']).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        
        # Initialize loss functions
        self.setup_loss_functions()
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(config['log_dir'])\n        \n        # Training state\n        self.global_step = 0\n        self.epoch = 0\n        self.best_val_loss = float('inf')\n        \n        logging.info(f\"Trainer initialized on device: {self.device}\")\n        logging.info(f\"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}\")\n    \n    def setup_logging(self):\n        \"\"\"Setup logging configuration.\"\"\"\n        log_dir = self.config['log_dir']\n        os.makedirs(log_dir, exist_ok=True)\n        \n        log_file = os.path.join(log_dir, f\"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log\")\n        \n        logging.basicConfig(\n            level=logging.INFO,\n            format='%(asctime)s - %(levelname)s - %(message)s',\n            handlers=[\n                logging.FileHandler(log_file),\n                logging.StreamHandler()\n            ]\n        )\n    \n    def setup_optimizer(self) -> optim.Optimizer:\n        \"\"\"Setup optimizer.\"\"\"\n        optimizer_config = self.config['optimizer']\n        \n        if optimizer_config['type'] == 'adam':\n            return optim.Adam(\n                self.model.parameters(),\n                lr=optimizer_config['lr'],\n                betas=optimizer_config.get('betas', (0.9, 0.999)),\n                weight_decay=optimizer_config.get('weight_decay', 1e-6)\n            )\n        elif optimizer_config['type'] == 'adamw':\n            return optim.AdamW(\n                self.model.parameters(),\n                lr=optimizer_config['lr'],\n                betas=optimizer_config.get('betas', (0.9, 0.999)),\n                weight_decay=optimizer_config.get('weight_decay', 1e-2)\n            )\n        else:\n            raise ValueError(f\"Unknown optimizer type: {optimizer_config['type']}\")\n    \n    def setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:\n        \"\"\"Setup learning rate scheduler.\"\"\"\n        scheduler_config = self.config.get('scheduler')\n        \n        if scheduler_config is None:\n            return None\n        \n        if scheduler_config['type'] == 'cosine':\n            return optim.lr_scheduler.CosineAnnealingLR(\n                self.optimizer,\n                T_max=scheduler_config['T_max'],\n                eta_min=scheduler_config.get('eta_min', 1e-8)\n            )\n        elif scheduler_config['type'] == 'step':\n            return optim.lr_scheduler.StepLR(\n                self.optimizer,\n                step_size=scheduler_config['step_size'],\n                gamma=scheduler_config.get('gamma', 0.1)\n            )\n        else:\n            raise ValueError(f\"Unknown scheduler type: {scheduler_config['type']}\")\n    \n    def setup_loss_functions(self):\n        \"\"\"Setup loss functions for training.\"\"\"\n        # Mel-spectrogram reconstruction loss\n        self.mel_loss = nn.L1Loss()\n        \n        # Audio reconstruction loss\n        self.audio_loss = nn.L1Loss()\n        \n        # Perceptual loss for voice over quality\n        self.perceptual_loss = PerceptualLoss()\n        \n        # Prosody loss\n        self.prosody_loss = nn.MSELoss()\n    \n    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:\n        \"\"\"Train for one epoch.\"\"\"\n        self.model.train()\n        epoch_losses = {'total': 0, 'mel': 0, 'audio': 0, 'perceptual': 0, 'prosody': 0}\n        \n        pbar = tqdm(train_loader, desc=f'Epoch {self.epoch}')\n        \n        for batch_idx, batch in enumerate(pbar):\n            # Move batch to device\n            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v \n                    for k, v in batch.items()}\n            \n            # Forward pass\n            self.optimizer.zero_grad()\n            \n            output = self.model(\n                text_input=batch['text_tokens'],\n                speaker_embedding=None  # Will be added later\n            )\n            \n            # Calculate losses\n            losses = self.calculate_losses(output, batch)\n            total_loss = losses['total']\n            \n            # Backward pass\n            total_loss.backward()\n            \n            # Gradient clipping\n            torch.nn.utils.clip_grad_norm_(self.model.parameters(), \n                                         self.config['training']['grad_clip_norm'])\n            \n            self.optimizer.step()\n            \n            # Update learning rate\n            if self.scheduler is not None:\n                self.scheduler.step()\n            \n            # Update metrics\n            for key, value in losses.items():\n                epoch_losses[key] += value.item()\n            \n            # Logging\n            if self.global_step % self.config['training']['log_interval'] == 0:\n                self.log_training_step(losses)\n            \n            # Update progress bar\n            pbar.set_postfix({\n                'loss': f\"{total_loss.item():.4f}\",\n                'lr': f\"{self.optimizer.param_groups[0]['lr']:.2e}\"\n            })\n            \n            self.global_step += 1\n        \n        # Average losses over epoch\n        for key in epoch_losses:\n            epoch_losses[key] /= len(train_loader)\n        \n        return epoch_losses\n    \n    def validate(self, val_loader: DataLoader) -> Dict[str, float]:\n        \"\"\"Validate model.\"\"\"\n        self.model.eval()\n        val_losses = {'total': 0, 'mel': 0, 'audio': 0, 'perceptual': 0, 'prosody': 0}\n        \n        with torch.no_grad():\n            for batch in tqdm(val_loader, desc='Validation'):\n                # Move batch to device\n                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v \n                        for k, v in batch.items()}\n                \n                # Forward pass\n                output = self.model(\n                    text_input=batch['text_tokens'],\n                    speaker_embedding=None\n                )\n                \n                # Calculate losses\n                losses = self.calculate_losses(output, batch)\n                \n                # Update metrics\n                for key, value in losses.items():\n                    val_losses[key] += value.item()\n        \n        # Average losses\n        for key in val_losses:\n            val_losses[key] /= len(val_loader)\n        \n        return val_losses\n    \n    def calculate_losses(self, output: Dict[str, torch.Tensor], \n                        batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:\n        \"\"\"Calculate training losses.\"\"\"\n        # Mel-spectrogram loss\n        mel_loss = self.mel_loss(output['mel_spectrogram'], \n                                batch['mel_spectrogram'])\n        \n        # Audio loss\n        audio_loss = self.audio_loss(output['audio'], \n                                   batch['audio'])\n        \n        # Perceptual loss for voice over quality\n        perceptual_loss = self.perceptual_loss(output['audio'], \n                                              batch['audio'])\n        \n        # Prosody loss (simplified)\n        prosody_loss = torch.tensor(0.0, device=self.device)  # Placeholder\n        \n        # Weighted total loss\n        weights = self.config['loss_weights']\n        total_loss = (weights['mel'] * mel_loss + \n                     weights['audio'] * audio_loss +\n                     weights['perceptual'] * perceptual_loss +\n                     weights['prosody'] * prosody_loss)\n        \n        return {\n            'total': total_loss,\n            'mel': mel_loss,\n            'audio': audio_loss,\n            'perceptual': perceptual_loss,\n            'prosody': prosody_loss\n        }\n    \n    def log_training_step(self, losses: Dict[str, torch.Tensor]):\n        \"\"\"Log training step metrics.\"\"\"\n        for key, value in losses.items():\n            self.writer.add_scalar(f'train/{key}_loss', value.item(), self.global_step)\n        \n        self.writer.add_scalar('train/learning_rate', \n                              self.optimizer.param_groups[0]['lr'], \n                              self.global_step)\n    \n    def log_epoch(self, train_losses: Dict[str, float], \n                  val_losses: Dict[str, float]):\n        \"\"\"Log epoch metrics.\"\"\"\n        # Log training losses\n        for key, value in train_losses.items():\n            self.writer.add_scalar(f'epoch/train_{key}_loss', value, self.epoch)\n        \n        # Log validation losses\n        for key, value in val_losses.items():\n            self.writer.add_scalar(f'epoch/val_{key}_loss', value, self.epoch)\n        \n        logging.info(f\"Epoch {self.epoch} - Train Loss: {train_losses['total']:.4f}, \"\n                    f\"Val Loss: {val_losses['total']:.4f}\")\n    \n    def save_checkpoint(self, val_loss: float, is_best: bool = False):\n        \"\"\"Save model checkpoint.\"\"\"\n        checkpoint = {\n            'epoch': self.epoch,\n            'global_step': self.global_step,\n            'model_state_dict': self.model.state_dict(),\n            'optimizer_state_dict': self.optimizer.state_dict(),\n            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,\n            'val_loss': val_loss,\n            'config': self.config\n        }\n        \n        # Save regular checkpoint\n        checkpoint_path = os.path.join(self.config['checkpoint_dir'], \n                                     f'checkpoint_epoch_{self.epoch}.pt')\n        torch.save(checkpoint, checkpoint_path)\n        \n        # Save best checkpoint\n        if is_best:\n            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')\n            torch.save(checkpoint, best_path)\n            logging.info(f\"New best model saved with validation loss: {val_loss:.4f}\")\n    \n    def load_checkpoint(self, checkpoint_path: str):\n        \"\"\"Load model checkpoint.\"\"\"\n        checkpoint = torch.load(checkpoint_path, map_location=self.device)\n        \n        self.model.load_state_dict(checkpoint['model_state_dict'])\n        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])\n        \n        if self.scheduler and checkpoint['scheduler_state_dict']:\n            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])\n        \n        self.epoch = checkpoint['epoch']\n        self.global_step = checkpoint['global_step']\n        self.best_val_loss = checkpoint['val_loss']\n        \n        logging.info(f\"Checkpoint loaded from epoch {self.epoch}\")\n    \n    def train(self, train_loader: DataLoader, val_loader: DataLoader):\n        \"\"\"Main training loop.\"\"\"\n        logging.info(\"Starting training...\")\n        \n        for epoch in range(self.epoch, self.config['training']['num_epochs']):\n            self.epoch = epoch\n            \n            # Train epoch\n            train_losses = self.train_epoch(train_loader)\n            \n            # Validate\n            val_losses = self.validate(val_loader)\n            \n            # Log metrics\n            self.log_epoch(train_losses, val_losses)\n            \n            # Save checkpoint\n            val_loss = val_losses['total']\n            is_best = val_loss < self.best_val_loss\n            \n            if is_best:\n                self.best_val_loss = val_loss\n            \n            if epoch % self.config['training']['save_interval'] == 0 or is_best:\n                self.save_checkpoint(val_loss, is_best)\n        \n        logging.info(\"Training completed!\")\n        self.writer.close()\n\n\nclass PerceptualLoss(nn.Module):\n    \"\"\"Perceptual loss for voice over quality assessment.\"\"\"\n    \n    def __init__(self):\n        super().__init__()\n        # Simplified perceptual loss - in practice, use pre-trained features\n        self.mse_loss = nn.MSELoss()\n    \n    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:\n        # Simplified implementation\n        return self.mse_loss(pred_audio, target_audio)\n\n\ndef create_training_config() -> Dict:\n    \"\"\"Create default training configuration.\"\"\"\n    return {\n        # Model configuration\n        'model': create_model_config(),\n        \n        # Data configuration\n        'data': {\n            'data_dir': '/path/to/egyptian_tts_data',\n            'sample_rate': 48000,\n            'bit_depth': 24,\n            'mel_dim': 80,\n            'n_fft': 2048,\n            'hop_length': 512,\n            'win_length': 2048,\n            'fmin': 0,\n            'fmax': 8000,\n            'use_augmentation': True\n        },\n        \n        # Training configuration\n        'training': {\n            'num_epochs': 1000,\n            'batch_size': 16,\n            'grad_clip_norm': 1.0,\n            'log_interval': 100,\n            'save_interval': 10\n        },\n        \n        # Optimizer configuration\n        'optimizer': {\n            'type': 'adamw',\n            'lr': 1e-4,\n            'betas': (0.9, 0.999),\n            'weight_decay': 1e-2\n        },\n        \n        # Scheduler configuration\n        'scheduler': {\n            'type': 'cosine',\n            'T_max': 1000,\n            'eta_min': 1e-8\n        },\n        \n        # Loss weights\n        'loss_weights': {\n            'mel': 1.0,\n            'audio': 0.5,\n            'perceptual': 0.3,\n            'prosody': 0.2\n        },\n        \n        # Directories\n        'log_dir': './logs',\n        'checkpoint_dir': './checkpoints'\n    }\n\n\ndef collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:\n    \"\"\"Custom collate function for variable length sequences.\"\"\"\n    # Pad sequences to same length\n    text_tokens = pad_sequence([item['text_tokens'] for item in batch], \n                              batch_first=True, padding_value=0)\n    phonemes = pad_sequence([item['phonemes'] for item in batch], \n                           batch_first=True, padding_value=0)\n    \n    # Handle mel-spectrograms (pad time dimension)\n    mel_specs = [item['mel_spectrogram'] for item in batch]\n    max_mel_len = max(mel.shape[1] for mel in mel_specs)\n    \n    padded_mels = []\n    for mel in mel_specs:\n        if mel.shape[1] < max_mel_len:\n            padding = torch.zeros(mel.shape[0], max_mel_len - mel.shape[1])\n            mel = torch.cat([mel, padding], dim=1)\n        padded_mels.append(mel)\n    \n    mel_spectrograms = torch.stack(padded_mels)\n    \n    # Handle audio (pad to same length)\n    audio_lengths = [item['audio'].shape[0] for item in batch]\n    max_audio_len = max(audio_lengths)\n    \n    padded_audio = []\n    for item in batch:\n        audio = item['audio']\n        if audio.shape[0] < max_audio_len:\n            padding = torch.zeros(max_audio_len - audio.shape[0])\n            audio = torch.cat([audio, padding])\n        padded_audio.append(audio)\n    \n    audio = torch.stack(padded_audio)\n    \n    return {\n        'text_tokens': text_tokens,\n        'phonemes': phonemes,\n        'mel_spectrogram': mel_spectrograms,\n        'audio': audio,\n        'speaker_ids': [item['speaker_id'] for item in batch]\n    }\n\n\ndef main():\n    \"\"\"Main training function.\"\"\"\n    parser = argparse.ArgumentParser(description='Train Egyptian Arabic TTS model')\n    parser.add_argument('--config', type=str, help='Path to config file')\n    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data')\n    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')\n    \n    args = parser.parse_args()\n    \n    # Load configuration\n    if args.config:\n        with open(args.config, 'r') as f:\n            config = json.load(f)\n    else:\n        config = create_training_config()\n    \n    # Update data directory\n    config['data']['data_dir'] = args.data_dir\n    \n    # Create directories\n    os.makedirs(config['log_dir'], exist_ok=True)\n    os.makedirs(config['checkpoint_dir'], exist_ok=True)\n    \n    # Create datasets\n    train_dataset = EgyptianTTSDataset(args.data_dir, config['data'], 'train')\n    val_dataset = EgyptianTTSDataset(args.data_dir, config['data'], 'val')\n    \n    # Create data loaders\n    train_loader = DataLoader(\n        train_dataset,\n        batch_size=config['training']['batch_size'],\n        shuffle=True,\n        collate_fn=collate_fn,\n        num_workers=4,\n        pin_memory=True\n    )\n    \n    val_loader = DataLoader(\n        val_dataset,\n        batch_size=config['training']['batch_size'],\n        shuffle=False,\n        collate_fn=collate_fn,\n        num_workers=4,\n        pin_memory=True\n    )\n    \n    # Initialize trainer\n    trainer = TTSTrainer(config)\n    \n    # Resume from checkpoint if specified\n    if args.resume:\n        trainer.load_checkpoint(args.resume)\n    \n    # Start training\n    trainer.train(train_loader, val_loader)\n\n\nif __name__ == \"__main__\":\n    main()

