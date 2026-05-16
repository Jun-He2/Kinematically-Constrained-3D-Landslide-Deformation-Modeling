import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfConsistentPhysicsLoss(nn.Module):
    def __init__(self, 
                 lambda_reg=1.0,     # weight for regression loss
                 lambda_dir=0.5,     # weight for direction supervision loss
                 lambda_cons=0.1,    # weight for self-consistency constraint loss
                 scaler=None,        # scaler object for inverse normalization
                 threshold_ratio=0.05): # relative threshold ratio for filtering stationary points
        super(SelfConsistentPhysicsLoss, self).__init__()
        
        self.lambda_reg = lambda_reg
        self.lambda_dir = lambda_dir
        self.lambda_cons = lambda_cons
        self.threshold_ratio = threshold_ratio
        
        # Register inverse normalization parameters
        # Buffers persist with the module but are not updated by the optimizer,
        # and they automatically follow the module's device.
        self.use_scaler = False
        if scaler is not None:
            self.use_scaler = True
            # Assume the first 3 columns of scaler's mean/scale correspond to x, y, z
            self.register_buffer('scale', torch.tensor(scaler.scale_[[0,1,2]], dtype=torch.float32))
            self.register_buffer('mean', torch.tensor(scaler.mean_[[0,1,2]], dtype=torch.float32))
            
        # Use SmoothL1Loss (Huber Loss) to prevent large gradients from destabilizing training
        self.reg_loss_fn = nn.SmoothL1Loss(beta=1.0)

    def forward(self, pred_disp, pred_ref_dir, target_disp, mask=None):
        """
        Args:
            pred_disp:    Predicted final displacement  [Batch, 3]
            pred_ref_dir: Predicted reference direction [Batch, 3] (already normalized)
            target_disp:  Ground truth displacement    [Batch, 3]
            mask:         (Optional) geometric validity mask [Batch, 1]
        """
        
        # 0. Basic data filtering (apply geometric mask if provided)
        if mask is not None:
            mask = mask.view(-1).bool()
            p_disp = pred_disp[mask]
            p_dir = pred_ref_dir[mask]
            t_disp = target_disp[mask]
        else:
            p_disp, p_dir, t_disp = pred_disp, pred_ref_dir, target_disp
            
        # Handle empty batch edge case
        if p_disp.shape[0] == 0:
            return torch.tensor(0.0, device=p_disp.device, requires_grad=True), {}

        # ==========================================================
        # Part 1: Regression Loss (Numerical Accuracy)
        # ==========================================================
        # Directly compute element-wise numerical error
        loss_reg = self.reg_loss_fn(p_disp, t_disp)

        # ==========================================================
        # Inverse Normalization (Critical!)
        # ==========================================================
        # Direction and projection calculations must be performed in real
        # physical space, because normalization distorts angles.
        if self.use_scaler:
            # Dynamically fetch device to avoid runtime errors
            device = p_disp.device
            scale = self.scale.to(device)
            # Note: if target is also normalized, both should be inverse-normalized here.
            # Assuming all inputs are normalized:
            # For computing vector differences/directions, the mean cancels out,
            # so multiplying by scale alone is sufficient.
            p_real = p_disp * scale # + mean
            t_real = t_disp * scale 
        else:
            p_real = p_disp
            t_real = t_disp

        # ==========================================================
        # Preparation: Compute magnitude and direction in real physical space
        # ==========================================================
        # Compute ground truth displacement magnitude
        t_magnitude = torch.norm(t_real, dim=1, keepdim=True) + 1e-9
        # Compute ground truth unit direction vector
        t_dir_unit = t_real / t_magnitude
        
        # Dynamic mask: only compute direction loss for points that are truly moving.
        # Threshold is set adaptively to 20% of the 95% maximum displacement in the current batch.

        k = max(1, int(t_magnitude.numel() * 0.95))
        top_magnitude = torch.kthvalue(t_magnitude.flatten(), k).values.detach()
        #move_threshold = self.threshold_ratio * t_magnitude.max().detach()
        move_threshold = self.threshold_ratio * top_magnitude
        move_mask = (t_magnitude > move_threshold).squeeze()
        
        # ==========================================================
        # Part 2: Direction Supervision Loss
        # ==========================================================
        # Supervise whether the output of "dir_head" is close to the true sliding direction
        loss_dir = torch.tensor(0.0, device=p_disp.device)
        
        if move_mask.sum() > 0:
            # Only consider points with significant displacement
            valid_p_dir = p_dir[move_mask]      # predicted direction
            valid_t_dir = t_dir_unit[move_mask] # ground truth direction
            
            # Cosine Loss: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(valid_p_dir, valid_t_dir, dim=1)
            loss_dir = torch.mean(1.0 - cos_sim)

        # ==========================================================
        # Part 3: Self-Consistency / Residual Loss
        # ==========================================================
        # Constraint: the predicted displacement p_real should lie primarily
        # along the predicted reference direction p_dir.
        # In other words, the component perpendicular to p_dir should be as small as possible.
        
        # Compute the projection of p_real onto p_dir: proj = (v · d) * d
        # p_dir is already a unit vector
        dot_prod = torch.sum(p_real * p_dir, dim=1, keepdim=True)
        projection = dot_prod * p_dir
        
        # Compute the residual (perpendicular component / lateral drift)
        residual = p_real - projection
        
        # Penalize the magnitude of the residual
        # Technique: penalize absolute residual magnitude with Huber Loss
        # to prevent large errors from dominating gradients.
        loss_cons = F.smooth_l1_loss(residual, torch.zeros_like(residual))

        # ==========================================================
        # Total Loss Aggregation
        # ==========================================================
        total_loss = (self.lambda_reg * loss_reg + 
                      self.lambda_dir * loss_dir + 
                      self.lambda_cons * loss_cons)
        
        # Return loss and monitoring dictionary
        loss_dict = {
            "reg_loss": loss_reg.item(),
            "dir_loss": loss_dir.item(),
            "cons_loss": loss_cons.item()
        }
        
        return total_loss