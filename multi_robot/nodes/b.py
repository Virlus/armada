def _load_policy(self):
        """Load policy from checkpoint"""
        payload = torch.load(open(self.eval_cfg.checkpoint_path, 'rb'), pickle_module=dill)
        self.cfg = payload['cfg']
        
        # Overwrite config values according to evaluation config
        self.cfg.policy.num_inference_steps = self.eval_cfg.policy.num_inference_steps
        self.cfg.output_dir = self.eval_cfg.output_dir
        if 'obs_encoder' in self.cfg.policy:
            self.cfg.policy.obs_encoder.pretrained_path = None
        
        # Initialize workspace
        import hydra
        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg, self.rank, self.world_size, self.device_id, self.device)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Get policy from workspace
        self.policy = workspace.model.module
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model.module
        
        self.policy.to(self.device)
        self.policy.eval()
        
        # Extract policy parameters
        self.To = self.policy.n_obs_steps
        self.Ta = self.policy.n_action_steps
        self.obs_feature_dim = self.policy.obs_feature_dim
        self.img_shape = self.cfg.task['shape_meta']['obs']['wrist_img']['shape']
        
        # Override Ta with evaluation config
        if hasattr(self.eval_cfg, 'Ta'):
            self.Ta = self.eval_cfg.Ta
    
def _setup_transformers(self):
    """Setup rotation transformers for action and observation spaces"""
    self.action_dim = self.cfg.shape_meta['action']['shape'][0]
    self.action_rot_transformer = None
    self.obs_rot_transformer = None
    
    # Check if there's need for transforming rotation representation
    if 'rotation_rep' in self.cfg.shape_meta['action']:
        self.action_rot_transformer = RotationTransformer(
            from_rep='quaternion', 
            to_rep=self.cfg.shape_meta['action']['rotation_rep']
        )
    
    if 'ee_pose' in self.cfg.shape_meta['obs']:
        self.ee_pose_dim = self.cfg.shape_meta['obs']['ee_pose']['shape'][0]
        self.state_type = 'ee_pose'
        self.state_shape = self.cfg.task['shape_meta']['obs']['ee_pose']['shape']
        if 'rotation_rep' in self.cfg.shape_meta['obs']['ee_pose']:
            self.obs_rot_transformer = RotationTransformer(
                from_rep='quaternion', 
                to_rep=self.cfg.shape_meta['obs']['ee_pose']['rotation_rep']
            )
    else:
        self.ee_pose_dim = self.cfg.shape_meta['obs']['qpos']['shape'][0]
        self.state_type = 'qpos'
        self.state_shape = self.cfg.task['shape_meta']['obs']['qpos']['shape']