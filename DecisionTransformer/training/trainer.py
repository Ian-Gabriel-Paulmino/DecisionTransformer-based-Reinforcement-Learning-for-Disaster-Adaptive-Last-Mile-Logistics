# """
# Training logic for Decision Transformer
# """
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from tqdm import tqdm
# import os
# from datetime import datetime


# class DecisionTransformerTrainer:
#     """
#     Trainer class for Decision Transformer
    
#     Handles:
#     - Training loop with validation
#     - Learning rate scheduling
#     - Gradient clipping
#     - Checkpoint saving
#     - TensorBoard logging
#     """
    
#     def __init__(self, model, config, train_loader, val_loader, 
#                  save_dir='checkpoints', device=None):
#         """
#         Initialize trainer
        
#         Args:
#             model: DecisionTransformer instance
#             config: DecisionTransformerConfig instance
#             train_loader: Training data loader
#             val_loader: Validation data loader
#             save_dir: Directory to save checkpoints (default: 'checkpoints')
#             device: torch device (default: auto-detect)
#         """
#         self.model = model
#         self.config = config
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.save_dir = save_dir
        
#         # Device
#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = device
        
#         print(f"Using device: {self.device}")
#         self.model.to(self.device)
        
#         # Optimizer (AdamW with weight decay)
#         self.optimizer = optim.AdamW(
#             model.parameters(),
#             lr=1e-4,
#             weight_decay=0.01,
#             betas=(0.9, 0.999)
#         )
        
#         # Learning rate scheduler (cosine annealing with warmup)
#         self.warmup_steps = 1000
#         self.total_steps = len(train_loader) * 100  # Assuming 100 epochs
#         self.scheduler = self._create_scheduler()
        
#         # Loss function (cross-entropy for action prediction)
#         self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
#         # Tracking
#         self.current_step = 0
#         self.best_val_loss = float('inf')
        
#         # TensorBoard logging
#         log_dir = f'runs/dt_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#         self.writer = SummaryWriter(log_dir)
#         print(f"TensorBoard logs: {log_dir}")
        
#         # Create save directory
#         os.makedirs(save_dir, exist_ok=True)
    
#     def _create_scheduler(self):
#         """
#         Create learning rate scheduler with warmup and cosine annealing
        
#         Returns:
#             LambdaLR scheduler
#         """
#         def lr_lambda(step):
#             # Warmup phase
#             if step < self.warmup_steps:
#                 return step / self.warmup_steps
#             # Cosine annealing phase
#             progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
#             return 0.5 * (1 + np.cos(np.pi * progress))
        
#         return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
#     def train_epoch(self, epoch):
#         """
#         Train for one epoch
        
#         Args:
#             epoch: Current epoch number
        
#         Returns:
#             tuple: (average_loss, average_accuracy)
#         """
#         self.model.train()
#         total_loss = 0
#         correct_predictions = 0
#         total_predictions = 0
        
#         pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
#         for batch_idx, batch in enumerate(pbar):
#             # Move batch to device
#             returns = batch['returns'].to(self.device)
#             states = batch['states'].to(self.device)
#             actions = batch['actions'].to(self.device)
#             timesteps = batch['timesteps'].to(self.device)
#             attention_mask = batch['attention_mask'].to(self.device)
            
#             # Forward pass
#             action_logits = self.model(
#                 returns, states, actions, timesteps, attention_mask
#             )
            
#             # Compute loss (next-token prediction)
#             # We predict action at t+1 given state at t
#             target_actions = actions[:, 1:].contiguous()
#             predicted_logits = action_logits[:, :-1, :].contiguous()
            
#             # Reshape for loss computation
#             loss = self.criterion(
#                 predicted_logits.view(-1, self.config.num_actions),
#                 target_actions.view(-1)
#             )
            
#             # Backward pass
#             self.optimizer.zero_grad()
#             loss.backward()
            
#             # Gradient clipping (prevent exploding gradients)
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
#             # Update weights
#             self.optimizer.step()
#             self.scheduler.step()
            
#             # Track metrics
#             total_loss += loss.item()
            
#             # Calculate accuracy
#             predictions = torch.argmax(predicted_logits, dim=-1)
#             correct = (predictions == target_actions).sum().item()
#             correct_predictions += correct
#             total_predictions += target_actions.numel()
            
#             # Update progress bar
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{correct / target_actions.numel():.3f}',
#                 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
#             })
            
#             # Log to TensorBoard
#             self.writer.add_scalar('Train/Loss', loss.item(), self.current_step)
#             self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.current_step)
            
#             self.current_step += 1
        
#         avg_loss = total_loss / len(self.train_loader)
#         avg_accuracy = correct_predictions / total_predictions
        
#         return avg_loss, avg_accuracy
    
#     def validate(self, epoch):
#         """
#         Validate the model
        
#         Args:
#             epoch: Current epoch number
        
#         Returns:
#             tuple: (average_loss, average_accuracy)
#         """
#         self.model.eval()
#         total_loss = 0
#         correct_predictions = 0
#         total_predictions = 0
        
#         pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
#         with torch.no_grad():
#             for batch in pbar:
#                 # Move batch to device
#                 returns = batch['returns'].to(self.device)
#                 states = batch['states'].to(self.device)
#                 actions = batch['actions'].to(self.device)
#                 timesteps = batch['timesteps'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
                
#                 # Forward pass
#                 action_logits = self.model(
#                     returns, states, actions, timesteps, attention_mask
#                 )
                
#                 # Compute loss
#                 target_actions = actions[:, 1:].contiguous()
#                 predicted_logits = action_logits[:, :-1, :].contiguous()
                
#                 loss = self.criterion(
#                     predicted_logits.view(-1, self.config.num_actions),
#                     target_actions.view(-1)
#                 )
                
#                 total_loss += loss.item()
                
#                 # Calculate accuracy
#                 predictions = torch.argmax(predicted_logits, dim=-1)
#                 correct = (predictions == target_actions).sum().item()
#                 correct_predictions += correct
#                 total_predictions += target_actions.numel()
                
#                 # Update progress bar
#                 pbar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{correct / target_actions.numel():.3f}'
#                 })
        
#         avg_loss = total_loss / len(self.val_loader)
#         avg_accuracy = correct_predictions / total_predictions
        
#         # Log to TensorBoard
#         self.writer.add_scalar('Val/Loss', avg_loss, epoch)
#         self.writer.add_scalar('Val/Accuracy', avg_accuracy, epoch)
        
#         return avg_loss, avg_accuracy
    
#     def save_checkpoint(self, epoch, val_loss, is_best=False):
#         """
#         Save model checkpoint
        
#         Args:
#             epoch: Current epoch number
#             val_loss: Validation loss
#             is_best: Whether this is the best model so far
#         """
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'val_loss': val_loss,
#             'config': self.config.__dict__
#         }
        
#         # Save regular checkpoint
#         path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
#         torch.save(checkpoint, path)
        
#         # Save best model
#         if is_best:
#             best_path = os.path.join(self.save_dir, 'best_model.pt')
#             torch.save(checkpoint, best_path)
#             print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
#     def train(self, num_epochs=100, save_every=10):
#         """
#         Main training loop
        
#         Args:
#             num_epochs: Number of epochs to train (default: 100)
#             save_every: Save checkpoint every N epochs (default: 10)
#         """
#         print(f"\n{'='*60}")
#         print(f"Starting training for {num_epochs} epochs")
#         print(f"{'='*60}\n")
        
#         for epoch in range(num_epochs):
#             # Train
#             train_loss, train_acc = self.train_epoch(epoch)
            
#             # Validate
#             val_loss, val_acc = self.validate(epoch)
            
#             # Print summary
#             print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
#             print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#             print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
#             # Save checkpoint
#             is_best = val_loss < self.best_val_loss
#             if is_best:
#                 self.best_val_loss = val_loss
            
#             if (epoch + 1) % save_every == 0 or is_best:
#                 self.save_checkpoint(epoch, val_loss, is_best)
            
#             print()
        
#         print(f"\n{'='*60}")
#         print(f"Training complete!")
#         print(f"Best validation loss: {self.best_val_loss:.4f}")
#         print(f"{'='*60}\n")
        
#         self.writer.close()

# """
# Decision Transformer Trainer - FULLY FIXED VERSION
# Optimized for limited data with proper device handling
# """
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# from tqdm import tqdm
# import os
# from datetime import datetime


# print("✓ Loading FIXED trainer.py - Version 2.0")


# class DecisionTransformerTrainer:
#     """
#     Enhanced trainer optimized for limited data
    
#     Key features:
#     - Proper device handling (no CPU/GPU mixing)
#     - Quality-weighted loss
#     - Return aspiration
#     - Early stopping
#     - Optimized for 6K+ samples
#     """
    
#     def __init__(self, model, config, train_loader, val_loader, 
#                  save_dir='checkpoints', device=None):
#         """Initialize trainer with proper device handling"""
        
#         print("\n" + "="*60)
#         print("Initializing Decision Transformer Trainer")
#         print("="*60)
        
#         self.model = model
#         self.config = config
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.save_dir = save_dir
        
#         # Device setup
#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = device
        
#         print(f"Device: {self.device}")
#         self.model.to(self.device)
        
#         # Optimizer - reduced LR for limited data
#         self.optimizer = optim.AdamW(
#             model.parameters(),
#             lr=3e-5,  # Conservative for small dataset
#             weight_decay=0.01,
#             betas=(0.9, 0.999)
#         )
#         print(f"Learning rate: 3e-5")
        
#         # Learning rate scheduler
#         self.warmup_steps = 200
#         self.total_steps = len(train_loader) * 100
#         self.scheduler = self._create_scheduler()
        
#         # Loss function
#         self.criterion = nn.CrossEntropyLoss(reduction='none')
        
#         # Tracking
#         self.current_step = 0
#         self.best_val_loss = float('inf')
#         self.patience = 20  # Early stopping
#         self.patience_counter = 0
#         self.min_delta = 0.001
        
#         # TensorBoard
#         log_dir = f'runs/dt_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
#         self.writer = SummaryWriter(log_dir)
#         print(f"TensorBoard: {log_dir}")
        
#         os.makedirs(save_dir, exist_ok=True)
#         print("="*60 + "\n")
    
#     def _create_scheduler(self):
#         """Create learning rate scheduler with warmup and cosine decay"""
#         def lr_lambda(step):
#             if step < self.warmup_steps:
#                 return step / self.warmup_steps
#             progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
#             return 0.5 * (1 + np.cos(np.pi * progress))
        
#         return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
#     def compute_trajectory_quality_weights(self, batch):
#         final_returns = batch['returns'][:, 0, 0]  # GPU
#         batch_size = final_returns.shape[0]
        
#         # Convert to Python scalars (avoids device issues)
#         min_val = final_returns.min().item()  # Python float
#         max_val = final_returns.max().item()  # Python float
        
#         if (max_val - min_val) > 1e-8:
#             # Create tensors explicitly on GPU
#             min_tensor = torch.tensor(min_val, device=self.device)
#             max_tensor = torch.tensor(max_val, device=self.device)
            
#             # Now all operations are GPU tensors
#             normalized = (final_returns - min_tensor) / (max_tensor - min_tensor + 1e-8)
#             offset = torch.tensor(0.5, device=self.device)
#             weights = normalized + offset
#         else:
#             # Explicitly create on GPU
#             weights = torch.ones(batch_size, device=self.device)
        
#         return weights  # Guaranteed GPU!
    
#     def compute_return_aspiration(self, returns_tensor):
#         """
#         Apply return aspiration - aim for better performance
        
#         Modifies returns to be 15-25% better (less negative = faster)
        
#         Args:
#             returns_tensor: (batch_size, seq_len, 1) on GPU
        
#         Returns:
#             aspirational_returns: Same shape, same device
#         """
#         improvement = np.random.uniform(0.75, 0.85)
#         return returns_tensor * improvement
    
#     def train_epoch(self, epoch, use_aspiration=True):
#         """
#         Train for one epoch with quality weighting
        
#         Args:
#             epoch: Current epoch number
#             use_aspiration: Whether to use return aspiration (default: True)
        
#         Returns:
#             (avg_loss, avg_accuracy): Tuple of metrics
#         """
#         self.model.train()
        
#         total_loss = 0.0
#         total_weighted_loss = 0.0
#         correct_predictions = 0
#         total_predictions = 0
        
#         pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
#         for batch_idx, batch in enumerate(pbar):
#             # Move batch to device
#             returns = batch['returns'].to(self.device)
#             states = batch['states'].to(self.device)
#             actions = batch['actions'].to(self.device)
#             timesteps = batch['timesteps'].to(self.device)
#             attention_mask = batch['attention_mask'].to(self.device)
            
#             batch_size = returns.shape[0]
            
#             # Apply return aspiration 50% of time
#             if use_aspiration and np.random.random() < 0.5:
#                 returns = self.compute_return_aspiration(returns)
            
#             # Forward pass
#             action_logits = self.model(
#                 returns, states, actions, timesteps, attention_mask
#             )
            
#             # Compute loss (predict next action)
#             target_actions = actions[:, 1:].contiguous()
#             predicted_logits = action_logits[:, :-1, :].contiguous()
            
#             # Cross-entropy loss per token
#             loss_per_token = self.criterion(
#                 predicted_logits.view(-1, self.config.num_actions),
#                 target_actions.view(-1)
#             )
#             loss_per_token = loss_per_token.view(batch_size, -1)
            
#             # Quality weighting
#             trajectory_weights = self.compute_trajectory_quality_weights(batch)
            
#             # Apply weights (broadcast across sequence)
#             weighted_loss = loss_per_token * trajectory_weights.unsqueeze(1)
#             loss = weighted_loss.mean()
            
#             # Backward pass
#             self.optimizer.zero_grad()
#             loss.backward()
            
#             # Gradient clipping for stability
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
#             # Update parameters
#             self.optimizer.step()
#             self.scheduler.step()
            
#             # Track metrics
#             total_loss += loss_per_token.mean().item()
#             total_weighted_loss += loss.item()
            
#             # Calculate accuracy
#             predictions = torch.argmax(predicted_logits, dim=-1)
#             correct = (predictions == target_actions).sum().item()
#             correct_predictions += correct
#             total_predictions += target_actions.numel()
            
#             # Update progress bar
#             batch_acc = correct / target_actions.numel()
#             pbar.set_postfix({
#                 'loss': f'{loss.item():.4f}',
#                 'acc': f'{batch_acc:.3f}',
#                 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
#             })
            
#             # TensorBoard logging
#             if self.current_step % 10 == 0:
#                 self.writer.add_scalar('Train/Loss', loss.item(), self.current_step)
#                 self.writer.add_scalar('Train/Accuracy', batch_acc, self.current_step)
#                 self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.current_step)
            
#             self.current_step += 1
        
#         # Epoch metrics
#         avg_loss = total_loss / len(self.train_loader)
#         avg_weighted_loss = total_weighted_loss / len(self.train_loader)
#         avg_accuracy = correct_predictions / total_predictions
        
#         return avg_weighted_loss, avg_accuracy
    
#     def validate(self, epoch):
#         """
#         Validate the model
        
#         Args:
#             epoch: Current epoch number
        
#         Returns:
#             (avg_loss, avg_accuracy): Tuple of validation metrics
#         """
#         self.model.eval()
        
#         total_loss = 0.0
#         correct_predictions = 0
#         total_predictions = 0
        
#         pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
#         with torch.no_grad():
#             for batch in pbar:
#                 # Move to device
#                 returns = batch['returns'].to(self.device)
#                 states = batch['states'].to(self.device)
#                 actions = batch['actions'].to(self.device)
#                 timesteps = batch['timesteps'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
                
#                 # Forward pass
#                 action_logits = self.model(
#                     returns, states, actions, timesteps, attention_mask
#                 )
                
#                 # Compute loss
#                 target_actions = actions[:, 1:].contiguous()
#                 predicted_logits = action_logits[:, :-1, :].contiguous()
                
#                 loss_per_token = self.criterion(
#                     predicted_logits.view(-1, self.config.num_actions),
#                     target_actions.view(-1)
#                 )
                
#                 loss = loss_per_token.mean()
#                 total_loss += loss.item()
                
#                 # Calculate accuracy
#                 predictions = torch.argmax(predicted_logits, dim=-1)
#                 correct = (predictions == target_actions).sum().item()
#                 correct_predictions += correct
#                 total_predictions += target_actions.numel()
                
#                 # Update progress bar
#                 pbar.set_postfix({
#                     'loss': f'{loss.item():.4f}',
#                     'acc': f'{correct / target_actions.numel():.3f}'
#                 })
        
#         # Validation metrics
#         avg_loss = total_loss / len(self.val_loader)
#         avg_accuracy = correct_predictions / total_predictions
        
#         # TensorBoard
#         self.writer.add_scalar('Val/Loss', avg_loss, epoch)
#         self.writer.add_scalar('Val/Accuracy', avg_accuracy, epoch)
        
#         return avg_loss, avg_accuracy
    
#     def save_checkpoint(self, epoch, val_loss, is_best=False):
#         """Save model checkpoint"""
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'scheduler_state_dict': self.scheduler.state_dict(),
#             'val_loss': val_loss,
#             'best_val_loss': self.best_val_loss,
#             'config': self.config.__dict__
#         }
        
#         # Save regular checkpoint
#         path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
#         torch.save(checkpoint, path)
        
#         # Save best model
#         if is_best:
#             best_path = os.path.join(self.save_dir, 'best_model.pt')
#             torch.save(checkpoint, best_path)
#             print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
#     def train(self, num_epochs=100, save_every=10):
#         """
#         Main training loop with early stopping
        
#         Args:
#             num_epochs: Maximum number of epochs
#             save_every: Save checkpoint every N epochs
#         """
#         print("\n" + "="*60)
#         print(f"Starting Training - {num_epochs} epochs max")
#         print("="*60)
#         print("Features:")
#         print("  ✓ Quality-weighted loss")
#         print("  ✓ Return aspiration (50% of batches)")
#         print("  ✓ Early stopping (patience=20)")
#         print("  ✓ Gradient clipping")
#         print("  ✓ Learning rate warmup + cosine decay")
#         print("="*60 + "\n")
        
#         for epoch in range(num_epochs):
#             # Train
#             train_loss, train_acc = self.train_epoch(epoch, use_aspiration=True)
            
#             # Validate
#             val_loss, val_acc = self.validate(epoch)
            
#             # Print summary
#             print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
#             print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#             print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
#             # Check for improvement
#             is_best = val_loss < (self.best_val_loss - self.min_delta)
            
#             if is_best:
#                 self.best_val_loss = val_loss
#                 self.patience_counter = 0
#                 print(f"  ✓ New best validation loss!")
#             else:
#                 self.patience_counter += 1
#                 print(f"  No improvement ({self.patience_counter}/{self.patience})")
            
#             # Save checkpoint
#             if (epoch + 1) % save_every == 0 or is_best:
#                 self.save_checkpoint(epoch, val_loss, is_best)
            
#             # Early stopping check
#             if self.patience_counter >= self.patience:
#                 print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
#                 print(f"   No improvement for {self.patience} consecutive epochs")
#                 break
            
#             # Check for overfitting
#             train_val_gap = abs(train_loss - val_loss)
#             if train_val_gap > 0.8:
#                 print(f"  ⚠ Large train-val gap ({train_val_gap:.3f}) - possible overfitting")
            
#             print()
        
#         # Training complete
#         print("\n" + "="*60)
#         print("Training Complete!")
#         print("="*60)
#         print(f"Best validation loss: {self.best_val_loss:.4f}")
#         print(f"Total epochs: {epoch+1}")
#         print(f"Checkpoints saved to: {self.save_dir}")
#         print("="*60 + "\n")
        
#         self.writer.close()



"""
Decision Transformer Trainer - SIMPLIFIED CPU VERSION
No GPU complications, just straightforward training
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime


print("✓ Loading SIMPLIFIED CPU trainer.py")


class DecisionTransformerTrainer:
    """
    Simplified trainer - CPU only, no quality weighting complications
    """
    
    def __init__(self, model, config, train_loader, val_loader, 
                 save_dir='checkpoints', device=None):
        """Initialize trainer"""
        
        print("\n" + "="*60)
        print("Initializing Decision Transformer Trainer (CPU Mode)")
        print("="*60)
        
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        
        # Force CPU to avoid device issues
        self.device = torch.device('cpu')
        
        print(f"Device: CPU (forced)")
        print(f"WARNING: Training will be slower on CPU")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=3e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        print(f"Learning rate: 3e-5")
        
        # Learning rate scheduler
        self.warmup_steps = 200
        self.total_steps = len(train_loader) * 100
        self.scheduler = self._create_scheduler()
        
        # Loss function - simple cross-entropy
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.patience = 20
        self.patience_counter = 0
        self.min_delta = 0.001
        
        # TensorBoard
        log_dir = f'runs/dt_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard: {log_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        print("="*60 + "\n")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """
        Train for one epoch - SIMPLIFIED
        No quality weighting, no aspiration, just standard training
        """
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to CPU (already there, but explicit)
            returns = batch['returns'].to(self.device)
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            timesteps = batch['timesteps'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            action_logits = self.model(
                returns, states, actions, timesteps, attention_mask
            )
            
            # Compute loss (predict next action)
            target_actions = actions[:, 1:].contiguous()
            predicted_logits = action_logits[:, :-1, :].contiguous()
            
            # Simple cross-entropy loss
            loss = self.criterion(
                predicted_logits.reshape(-1, self.config.num_actions),
                target_actions.reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(predicted_logits, dim=-1)
            correct = (predictions == target_actions).sum().item()
            correct_predictions += correct
            total_predictions += target_actions.numel()
            
            # Update progress bar
            batch_acc = correct / target_actions.numel()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.3f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # TensorBoard logging
            if self.current_step % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.current_step)
                self.writer.add_scalar('Train/Accuracy', batch_acc, self.current_step)
                self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.current_step)
            
            self.current_step += 1
        
        # Epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = correct_predictions / total_predictions
        
        return avg_loss, avg_accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # Move to CPU
                returns = batch['returns'].to(self.device)
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                timesteps = batch['timesteps'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                action_logits = self.model(
                    returns, states, actions, timesteps, attention_mask
                )
                
                # Compute loss
                target_actions = actions[:, 1:].contiguous()
                predicted_logits = action_logits[:, :-1, :].contiguous()
                
                loss = self.criterion(
                    predicted_logits.reshape(-1, self.config.num_actions),
                    target_actions.reshape(-1)
                )
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(predicted_logits, dim=-1)
                correct = (predictions == target_actions).sum().item()
                correct_predictions += correct
                total_predictions += target_actions.numel()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / target_actions.numel():.3f}'
                })
        
        # Validation metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = correct_predictions / total_predictions
        
        # TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/Accuracy', avg_accuracy, epoch)
        
        return avg_loss, avg_accuracy
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    def train(self, num_epochs=100, save_every=10):
        """
        Main training loop with early stopping
        """
        print("\n" + "="*60)
        print(f"Starting Training - {num_epochs} epochs max")
        print("="*60)
        print("Mode: CPU only (simplified)")
        print("Features:")
        print("  ✓ Standard cross-entropy loss")
        print("  ✓ Early stopping (patience=20)")
        print("  ✓ Gradient clipping")
        print("  ✓ Learning rate warmup + cosine decay")
        print("="*60 + "\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Check for improvement
            is_best = val_loss < (self.best_val_loss - self.min_delta)
            
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  ✓ New best validation loss!")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.patience})")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break
            
            # Overfitting check
            train_val_gap = abs(train_loss - val_loss)
            if train_val_gap > 0.8:
                print(f"  ⚠ Large train-val gap ({train_val_gap:.3f})")
            
            print()
        
        # Complete
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total epochs: {epoch+1}")
        print("="*60 + "\n")
        
        self.writer.close()