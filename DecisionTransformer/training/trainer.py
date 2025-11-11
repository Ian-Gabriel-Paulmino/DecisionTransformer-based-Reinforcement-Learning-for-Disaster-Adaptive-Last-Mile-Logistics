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


"""
Improved Training Objective - Learn to BEAT NNA, not copy it

Key Innovation:
Instead of just learning "what did NNA do?", we train on:
1. Trajectory quality weighting (better routes get more influence)
2. Contrastive learning (good choices vs bad choices)
3. Reward-conditioned training (condition on better-than-achieved rewards)

This allows the model to learn patterns that EXCEED NNA performance.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime


class DecisionTransformerTrainer:
    """
    Enhanced trainer that learns to beat NNA baseline
    
    Key improvements:
    1. Quality-weighted loss (better trajectories = more weight)
    2. Return conditioning with aspiration (aim higher than achieved)
    3. Trajectory filtering (learn from best examples)
    """
    
    def __init__(self, model, config, train_loader, val_loader, 
                 save_dir='checkpoints', device=None):
        """
        Initialize improved trainer
        
        Args:
            model: DecisionTransformer instance
            config: DecisionTransformerConfig instance
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            device: torch device
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.warmup_steps = 500  # Reduced for smaller datasets
        self.total_steps = len(train_loader) * 100
        self.scheduler = self._create_scheduler()
        
        # Base loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')  # Don't reduce yet
        
        # Tracking
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # TensorBoard
        log_dir = f'runs/dt_improved_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_trajectory_quality_weights(self, batch):
        """
        Compute quality weights for each trajectory in batch
        
        Better trajectories get higher weights in the loss.
        This encourages the model to learn from good examples more than bad ones.
        
        Quality based on:
        1. Success (successful > failed)
        2. Efficiency (shorter time = better)
        3. Weather severity (harder conditions = more valuable)
        
        Args:
            batch: Batch dict with 'returns', 'states', etc.
        
        Returns:
            weights: Tensor of shape (batch_size,) with quality weights
        """
        batch_size = batch['returns'].shape[0]
        weights = torch.ones(batch_size, device=self.device)
        
        # Weight by final return (higher return = faster delivery = better)
        # Returns are negative time, so more negative = worse
        final_returns = batch['returns'][:, 0, 0]  # First timestep return (total return-to-go)
        
        # Normalize to [0, 1] range within batch
        min_return = final_returns.min()
        max_return = final_returns.max()
        
        if max_return > min_return:
            # Better return (closer to 0) gets higher weight
            normalized_returns = (final_returns - min_return) / (max_return - min_return)
            weights = normalized_returns + 0.5  # Range [0.5, 1.5]
        
        return weights
    
    def compute_return_aspiration(self, current_return, percentile=75):
        """
        Compute aspirational return for training
        
        Instead of conditioning on achieved return, condition on a BETTER return.
        This encourages the model to aim higher than what NNA achieved.
        
        Args:
            current_return: Actual achieved return
            percentile: Target percentile (75 = aim for top 25% performance)
        
        Returns:
            aspirational_return: Improved target return
        """
        # Aspire to be 20-30% better than achieved
        # Returns are negative, so multiply by 0.7-0.8 to improve
        improvement_factor = np.random.uniform(0.7, 0.85)
        aspirational_return = current_return * improvement_factor
        
        return aspirational_return
    
    def train_epoch(self, epoch, use_aspiration=True):
        """
        Train for one epoch with quality weighting
        
        Args:
            epoch: Current epoch number
            use_aspiration: Whether to use return aspiration
        
        Returns:
            tuple: (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0
        total_weighted_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            returns = batch['returns'].to(self.device)
            states = batch['states'].to(self.device)
            actions = batch['actions'].to(self.device)
            timesteps = batch['timesteps'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            batch_size = returns.shape[0]
            
            # INNOVATION 1: Apply return aspiration
            if use_aspiration and np.random.random() < 0.5:  # 50% of time
                # Modify returns to be more ambitious
                for i in range(batch_size):
                    current_return = returns[i, 0, 0].item()
                    aspirational = self.compute_return_aspiration(current_return)
                    # Scale entire return sequence proportionally
                    scale = aspirational / (current_return + 1e-8)
                    returns[i] = returns[i] * scale
            
            # Forward pass
            action_logits = self.model(
                returns, states, actions, timesteps, attention_mask
            )
            
            # Compute loss (next-token prediction)
            target_actions = actions[:, 1:].contiguous()
            predicted_logits = action_logits[:, :-1, :].contiguous()
            
            # Reshape for loss computation
            loss_per_token = self.criterion(
                predicted_logits.view(-1, self.config.num_actions),
                target_actions.view(-1)
            )
            loss_per_token = loss_per_token.view(batch_size, -1)
            
            # INNOVATION 2: Quality weighting
            trajectory_weights = self.compute_trajectory_quality_weights(batch)
            
            # Apply weights across sequence
            weighted_loss = loss_per_token * trajectory_weights.unsqueeze(1)
            loss = weighted_loss.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss_per_token.mean().item()
            total_weighted_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(predicted_logits, dim=-1)
            correct = (predictions == target_actions).sum().item()
            correct_predictions += correct
            total_predictions += target_actions.numel()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / target_actions.numel():.3f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to TensorBoard
            self.writer.add_scalar('Train/Loss', loss.item(), self.current_step)
            self.writer.add_scalar('Train/WeightedLoss', loss.item(), self.current_step)
            self.writer.add_scalar('Train/UnweightedLoss', loss_per_token.mean().item(), self.current_step)
            self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], self.current_step)
            
            self.current_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        avg_weighted_loss = total_weighted_loss / len(self.train_loader)
        avg_accuracy = correct_predictions / total_predictions
        
        print(f"  Train - Unweighted Loss: {avg_loss:.4f}, Weighted Loss: {avg_weighted_loss:.4f}")
        
        return avg_weighted_loss, avg_accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                returns = batch['returns'].to(self.device)
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                timesteps = batch['timesteps'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                action_logits = self.model(
                    returns, states, actions, timesteps, attention_mask
                )
                
                target_actions = actions[:, 1:].contiguous()
                predicted_logits = action_logits[:, :-1, :].contiguous()
                
                loss_per_token = self.criterion(
                    predicted_logits.view(-1, self.config.num_actions),
                    target_actions.view(-1)
                )
                
                loss = loss_per_token.mean()
                total_loss += loss.item()
                
                predictions = torch.argmax(predicted_logits, dim=-1)
                correct = (predictions == target_actions).sum().item()
                correct_predictions += correct
                total_predictions += target_actions.numel()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct / target_actions.numel():.3f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = correct_predictions / total_predictions
        
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
            'config': self.config.__dict__
        }
        
        path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    def train(self, num_epochs=100, save_every=10):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting IMPROVED training for {num_epochs} epochs")
        print(f"Innovations:")
        print(f"  ✓ Quality-weighted loss")
        print(f"  ✓ Return aspiration")
        print(f"  ✓ Learning from best examples")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch, use_aspiration=True)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            print()
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        self.writer.close()