EXPERIMENTS = {
    'cifar10': {
        # These parameters should only have one value in the grid. These are
        # common for the entire experiment. These will not be used to compare runs
        # when testing if a run already exists.
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/3.0},
                'world_size': 1,
                'num_workers': 12,
            },
            # skip, recreate, continue
            'dedup_policy': 'skip',
        },
        'gridgen': 'resnetall',
        'dataflow': {'ds_name': 'CIFAR10'},
        'input_cfg': {'input_shape': (3, 32, 32)},
        # We handle gridding of the following
        'trial': [0, 1, 2],
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

