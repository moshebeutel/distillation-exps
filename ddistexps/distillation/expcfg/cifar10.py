EXPERIMENTS = {
    'distil-cifar10': {
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/2.0},
                'world_size': 1,
                'num_workers': 8,
            },
            'gridgen': 'resnetgen',
            'dedup_policy': 'version',  # 'ignore' or 'version' (default)
        },
        'dataflow': {
            'data_set': 'CIFAR10',
            'read_parallelism': 128,
        },
        'dispatch': 'distillation',
        'ckpt_file': None,
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 128},
        'trunk_cfg': {
            'name': ['CIFAR10ResNet56', 'ClipCIFAR10'],
        },
        'train_cfg': {
            'num_epochs': [200],
            'batch_size_gpu': 128,
            'optim':[
                {'name': 'adam', 'lr': 0.01, 'lr_scheduler': None,},
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.5, 0.75]},
                # {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                #  'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                #  'lr_milestone_fracs': [.5, 0.75, 0.9]},
                # {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                #  'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                #  'lr_milestone_fracs': [.4, 0.6]},
                # {'name': 'adam', 'lr': 0.005, 'lr_scheduler': 'cosine',},
            ],
            'transform': {
                'global_shuffle': True,
            },
            'use_amp': [True, False],
            'loss_cfg': {
                'distil_reg': [0.5], 'xentropy_reg': [0.5],
                'temperature': [
                    # { 'value': 1.0, 'gamma': 1.0, 'milestone_fracs': [1.0],},
                    { 'value': .3, 'gamma': 1.0, 'milestone_fracs': [0.5, 0.9],},
                    { 'value': 4.0, 'gamma': 0.5, 'milestone_fracs': [0.5, 9.0],},
                ],
            },
        },
    }
}
