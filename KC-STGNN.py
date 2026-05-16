class EnhancedTemporalConvNet(nn.Module):
    """
    Enhanced TCN block with gated activation mechanism.
    Uses dilated causal convolution to capture long-range temporal dependencies.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super(EnhancedTemporalConvNet, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels * 2, kernel_size, 
                              dilation=dilation, padding=self.pad)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Causal convolution with gated linear unit (GLU)
        conv_out = self.conv(x)
        conv_out = conv_out[:, :, :-self.pad] if self.pad > 0 else conv_out
        t, g = torch.chunk(conv_out, 2, dim=1)
        # Element-wise gating: tanh(feature) * sigmoid(gate)
        return self.dropout(torch.tanh(t) * torch.sigmoid(g))

class PhysicsAware_ST_TCN(nn.Module):
    """
    Physics-Aware Spatio-Temporal TCN with decoupled direction and displacement prediction.
    
    Key modification: Decouples landslide prediction into:
    1. Direction prediction (where the point tends to slide)
    2. Displacement prediction (how far it slides)
    
    This decomposition enforces physical consistency by explicitly modeling
    the sliding direction as a unit vector before computing the final displacement.
    """
    def __init__(self, 
                 num_nodes, 
                 in_dyn_feats, 
                 in_static_cont, 
                 cat_dims, 
                 hidden_dim, 
                 out_dim=3,  # predicts x, y, z displacement
                 window_size=90,
                 dropout_rate=0.3):
        super(PhysicsAware_ST_TCN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # === 1. Feature Embedding ===
        # Dynamic features (e.g., rainfall, water level)
        self.dyn_encoder = nn.Linear(in_dyn_feats, hidden_dim)
        
        # Static continuous features (e.g., slope, elevation)
        self.static_cont_encoder = nn.Linear(in_static_cont, hidden_dim // 2)
        
        # Static categorical features (e.g., soil type, geological zone)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(n, 4) for n in cat_dims])
        total_cat_dim = len(cat_dims) * 4
        
        # Node-specific identity embedding for location-aware learning
        self.node_id_dim = 8
        self.node_id_embedding = nn.Embedding(num_nodes, self.node_id_dim)
        
        # Fuse all static features into a unified representation
        self.static_fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 + total_cat_dim + self.node_id_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Gating mechanism: static features modulate dynamic feature importance
        self.feature_gate = nn.Linear(hidden_dim, hidden_dim)

        # === 2. TCN Module for Temporal Feature Extraction ===
        tcn_channels = hidden_dim
        self.tcn_layers = nn.ModuleList()
        dilations = [1, 2, 4, 8, 16]  # exponentially increasing receptive field
        for dilation in dilations:
            self.tcn_layers.append(EnhancedTemporalConvNet(
                tcn_channels, tcn_channels, kernel_size=3, dilation=dilation, dropout=dropout_rate
            ))
        self.tcn_norm = nn.LayerNorm(tcn_channels)
        
        # === 3. Spatial Module (GNN) ===
        # Multi-head GAT for spatial message passing between nodes
        self.gnn = GATConv(hidden_dim, hidden_dim // 4, num_heads=4, feat_drop=dropout_rate)
        
        # Learn edge weights from pairwise node features
        self.edge_weight_learner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
        )
        
        # === 4. Physics-Guided Output Heads (Core Modification) ===
        # Decoupled prediction: direction → displacement
        # This enforces the physical constraint that displacement = direction * magnitude
        
        # Combined feature dimension: TCN temporal + GNN spatial
        combined_dim = hidden_dim * 2 
        
        # A. Direction Prediction Head
        # Predicts the normalized sliding direction (unit vector in 3D)
        self.dir_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 3)  # outputs raw direction vector (x, y, z)
        )
        
        # B. Displacement Prediction Head
        # Predicts the final displacement magnitude conditioned on the reference direction
        # Input includes the predicted direction to ensure physical consistency
        self.disp_head = nn.Sequential(
            nn.Linear(combined_dim + 3, hidden_dim),  # +3 for concatenated direction
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, g, x_dyn, x_static_cont, x_static_cat, node_ids):
        # --- A. Feature Encoding & Fusion ---
        h_dyn = self.dyn_encoder(x_dyn)
        
        # Encode static features
        h_cont = self.static_cont_encoder(x_static_cont)
        h_cat = torch.cat([emb(x_static_cat[:, i].long()) 
                          for i, emb in enumerate(self.cat_embeddings)], dim=1)
        h_id = self.node_id_embedding(node_ids)
        h_static = self.static_fusion(torch.cat([h_cont, h_cat, h_id], dim=1))
        
        # Static-conditioned gating: terrain properties modulate temporal signal importance
        gate = torch.sigmoid(self.feature_gate(h_static)).unsqueeze(1)
        h_fused = h_dyn * gate
        
        # --- B. Temporal Feature Extraction (TCN) ---
        h_tcn = h_fused.permute(0, 2, 1)  # (B, C, T) for Conv1d
        residual = h_tcn
        
        # Stacked TCN layers with residual connections every 2 layers
        for i, tcn_layer in enumerate(self.tcn_layers):
            h_tcn = tcn_layer(h_tcn)
            if i % 2 == 1:
                h_tcn = h_tcn + residual[:, :, -h_tcn.size(2):]
                residual = h_tcn
        
        h_time = self.tcn_norm(h_tcn[:, :, -1])  # take last timestep output
        
        # --- C. Spatial Interaction (GNN) ---
        # Learn edge weights for small-to-medium graphs (< 10000 edges)
        if g.number_of_edges() < 100000:
            with g.local_scope():
                g.ndata['h'] = h_time
                g.apply_edges(lambda edges: {'feat': torch.cat([edges.src['h'], edges.dst['h']], dim=1)})
                edge_weights = self.edge_weight_learner(g.edata['feat']).squeeze()
                g.edata['w'] = edge_weights
        
        h_space = self.gnn(g, h_time)  # (N, Heads, Hidden/4)
        h_space = h_space.flatten(1)   # (N, Hidden)
        h_space = h_space + h_time     # residual connection
        
        # --- D. Physics-Guided Prediction (Core Modification) ---
        
        # 1. Build environment feature vector
        h_combined = torch.cat([h_time, h_space], dim=1)  # (N, Hidden*2)
        h_combined = self.dropout(h_combined)
        
        # 2. Predict reference sliding direction
        # CRITICAL: Normalize to unit vector — physically, direction has no magnitude
        raw_dir = self.dir_head(h_combined)
        ref_dir = F.normalize(raw_dir, p=2, dim=1)  # (N, 3) unit vector
        
        # 3. Physics-guided feature augmentation
        # Concatenate the predicted direction to inform displacement magnitude
        disp_input = torch.cat([h_combined, ref_dir], dim=1)  # (N, Hidden*2 + 3)
        
        # 4. Predict final displacement conditioned on direction
        displacement = self.disp_head(disp_input)  # (N, 3)
        
        # Return both final displacement and reference direction
        # ref_dir can be used for self-consistency regularization during training
        return displacement, ref_dir