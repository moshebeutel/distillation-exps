EXPERIMENTS = {
    'debug': {
        # These parameters should only have one value in the grid. These are
        # common for the entire experiment.
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/3.0},
                'world_size': 1,
                'num_workers': 6,
            },
            'dedup_policy': 'version',  # 'ignore' or 'version' (default)
            'src_exp_names': [['baseline/cifar10', 'distillation/cifar10']],
        },
        'dataflow': {
            # We will use the augmented dataflow on top of this. (Changes preprocessing)
            'data_set': 'TinyCIFAR10',
            'read_parallelism': 128,
        },
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 512},
        'trunk_cfg': [
            { 'expname': 'baseline/cifar10', 'runname': 'debonair-ray-624'},
            {'legacy_name': 'ClipCIFAR10'}, 
        ],
        'train_cfg': {
            'num_epochs': [2],
            'batch_size_gpu': 128,
            'optim':[
                {'name': 'adam', 'lr': 0.01, 'lr_scheduler': None,},
            ],
            'transform': { 'global_shuffle': True, },
            'use_amp': False,
            'loss_cfg': {
                'distil_reg': [1.0,], 'xentropy_reg': [1.0],
                'temperature': [
                    { 'value': .3, 'gamma': 1.0, 'milestone_fracs': [0.5, 0.9],},
                ],
            },
        },
    }
}
