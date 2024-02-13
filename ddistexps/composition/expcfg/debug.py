EXPERIMENTS = {
    'debug': {
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/3.0},
                'world_size': 1,
                'num_workers': 3,
            },
            'gridgen': 'resnetv3patchedgen',
            'dedup_policy': 'ignore',  # 'ignore' or 'version' (default)
        },
        # Following can be gridded
        'dataflow': { 'data_set': 'TinyCIFAR10', 'read_parallelism': 128, },
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 512},
        'trunk_cfg': [
            { 'expname': 'baseline/cifar10', 'runname': 'debonair-ray-624'},
        ],
        'compose_cfg':{
            'src_run_cfg': [
                {'expname': 'baseline/cifar10', 'runname': 'glamorous-foal-579',},
                {'expname': 'distillation/cifar10', 'runname': 'bittersweet-mare-631',},
            ],
            'conn_name': [ 'layer_by_layer' ],
        },
        'train_cfg': {
            'num_epochs': [3],
            'batch_size_gpu': 128,
            'optim':[
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.5, 0.75]},
            ],
            'transform': { 'global_shuffle': True, },
            'use_amp': False,
            'loss_cfg': {
                'distil_reg': [0.0,], 'xentropy_reg': [1.0],
                'temperature': [
                    { 'value': .3, 'gamma': 1.0, 'milestone_fracs': [0.5, 0.9],},
                ],
            },
        },
    }
}
