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
            { 'run_id': '424fda1257a04036ac3283dfa3a477d1'},
        ],
        'compose_cfg':{
            # Specify source runids
            'src_run_id': ['3285aa30c94b46eb941812c3ded6911d'],
            'conn_name': ['layer_by_layer', 'noconn'],
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
