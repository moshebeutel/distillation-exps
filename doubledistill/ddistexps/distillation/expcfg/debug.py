EXPERIMENTS = {
    'debug': {
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/2.0},
                'world_size': 1,
                'num_workers': 2,
            },
            'gridgen': 'resnetgen',
            'dedup_policy': 'version',  # 'ignore' or 'version' (default)
        },
        # Following can be gridded
        'dataflow': {
            'data_set': 'TinyCIFAR10',
            'read_parallelism': 128,
        },
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 512},
        'trunk_cfg': [
            { 'expname': 'baseline/cifar10', 'runname': 'adorable-snipe-986'}, # 44
            { 'expname': 'baseline/cifar10', 'runname': 'intrigued-hare-807'}, # 110
        ],
        'train_cfg': {
            'num_epochs': [2],
            'batch_size_gpu': 128,
            'optim':[
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.5, 0.75]},
            ],
            'transform': {
                # TODO: Test this
                'global_shuffle': True,
            },
            'use_amp': False,
            'loss_cfg': {
                'distil_reg': [1.0,], 'xentropy_reg': [1.0],
                'temperature': [
                    { 'value': 1.0, 'gamma': 1.0, 'milestone_fracs': [1.0],},
                ],
            },
        },
    }
}
