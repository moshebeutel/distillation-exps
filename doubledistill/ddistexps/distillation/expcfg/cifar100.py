import os

OUTDIR = os.path.join(os.environ['DDIST_EXPS_DIR'])

EXPERIMENTS = {
    'baseline-cifar100': {
        # These will not be used for comparison
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 0.5},
                'world_size': 1,
                'num_workers': 4,
            },
            'gridgen': 'resnetgen',
            'dedup_policy': 'ignore',  # 'ignore' or 'version' (default)
        },
        'dataflow': {
            'data_set': 'CIFAR100',
            'preprocessor':  'net',
            'read_parallelism': 128,
        },
        'dispatch': ['BLDispatch'],
        'ckpt_file': None,
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 512},
        'train_cfg': {
            'num_epochs': [200, 400],
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
                'global_shuffle': [True, False],
            },
            'use_amp': False,
            'lossfn': 'xentropy',
        },
    }

}


