EXPERIMENTS = {
    'cifar10': {
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/3.0},
                'world_size': 1,
                'num_workers': 6,
            },
            'gridgen': 'resnetv3patchedgen',
            'dedup_policy': 'ignore',  # 'ignore' or 'version' (default)
        },
        'trial': [0, 1, 2],
        'dataflow': {
            'data_set': 'CIFAR10',
            'read_parallelism': 128,
            'preprocessor':  'net',
        },''
        'dispatch': 'BLDispatch',
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 512},
        'train_cfg': {
            'num_epochs': [90],
            'batch_size_gpu': 128,
            'optim':[
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [0.3, .6, 0.9]},
            ],
            'transform': { 'global_shuffle': True,},
            'use_amp': False,
            'lossfn': 'xentropy',
        },
    }
}

