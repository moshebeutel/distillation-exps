EXPERIMENTS = {
    'debug': {
        # These parameters should only have one value in the grid. These are
        # common for the entire experiment.
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/4.0},
                'world_size': 1,
                'num_workers': 8,
            },
            'gridgen': 'resnetgen',
            'dedup_policy': 'version',  # 'ignore' or 'version' (default)
        },
        # Following can be gridded
        'dataflow': {
            'data_set': 'TinyCIFAR10',
            'read_parallelism': 128,
            'preprocessor':  'net',
        },
        'dispatch': ['distillation', 'baseline'],
        'ckpt_file': None,
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 512},
        'train_cfg': {
            'num_epochs': [10],
            'batch_size_gpu': 128,
            'optim':[
                {'name': 'adam', 'lr': 0.01, 'lr_scheduler': None,},
                {'name': 'adam', 'lr': 0.005, 'lr_scheduler': 'cosine',},
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.5, 0.75]},
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.5, 0.75, 0.9]},
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.4, 0.6]},
            ],
            'transform': {
                'global_shuffle': True,
            },
            'distil_reg': 1.0,
            'xentropy_reg': 1.0,
            'use_amp': False,
            'temperature': [
                { 'gamma': 1.0, 'milestone_fracs': [1.0],},
            ],
            'lossfn': 'xentropy', # Only used by baseline-trianer
        },
    }
}
