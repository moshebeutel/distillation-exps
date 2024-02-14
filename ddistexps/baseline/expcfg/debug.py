EXPERIMENTS = {
    'debug': {
        # These parameters should only have one value in the grid. These are
        # common for the entire experiment. These will not be used to compare runs
        # when testing if a run already exists.
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0},
                'world_size': 2,
                'num_workers': 4,
            },
            # skip, recreate, continue
            'dedup_policy': 'recreate',  
        },
        'gridgen': 'resnetv3patchedgen',
        'dataflow': {'ds_name': 'TinyCIFAR10', },
        'input_cfg': {'input_shape': (3, 32, 32)},
        # We handle gridding of the following
        'test_cfg': {'batch_size_gpu': 512},
        'train_cfg': {
            'num_epochs': [1],
            'batch_size_gpu': 32,
            'optim':[
                {'name': 'sgd', 'lr': 0.01, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.5, 0.75, 0.9]},
            ],
            'transform': { 'global_shuffle': True,},
            'use_amp': False,
            'lossfn': 'xentropy',
        },
    }
}
