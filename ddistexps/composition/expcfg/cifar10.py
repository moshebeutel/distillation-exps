EXPERIMENTS = {
    'cifar10': {
        'meta': {
            'worker_cfg': {
                'resource_req': {'num_gpus': 1.0/3.0},
                'world_size': 1,
                'num_workers': 3,
            },
            'gridgen': 'resnetv3patchedgen',
            'dedup_policy': 'ignore',  # 'ignore' or 'version' (default)
        },
        'dataflow': { 'data_set': 'CIFAR10', 'read_parallelism': 128, },
        'input_cfg': {'input_shape': (3, 32, 32)},
        'test_cfg': {'batch_size_gpu': 128},
        'trunk_cfg': {
            'run_id': '424fda1257a04036ac3283dfa3a477d1', # baseline/cf10 44, 91%
        },          
        'compose_cfg': {
            'src_run_id': [
                '3285aa30c94b46eb941812c3ded6911d', # 56%, Residual CNN
            ],
            'conn_name': [
                # 'noconn', 'residual_error', 'share_all',
                # 'share_post_layer', 'resnet_conn',
                'layer_by_layer', 'noconn',
            ],
        },
        'trial': [0, 1, 2],
        'train_cfg': {
            'num_epochs': [100],
            'batch_size_gpu': 128,
            'optim':[
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-4, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.3, 0.6, 0.9]},
                {'name': 'sgd', 'lr': 0.1, 'lr_scheduler': 'multistep',
                 'momentum': 0.9, 'weight_decay': 5e-3, 'lr_gamma': 0.2,
                 'lr_milestone_fracs': [.3, 0.6, 0.9]},
            ],
            'transform': { 'global_shuffle': True, },
            'use_amp': False,
            'loss_cfg': {
                'distil_reg': [1.0], 'xentropy_reg': [1.0],
                'temperature': [
                    { 'value': .3, 'gamma': 1.0, 'milestone_fracs': [0.5, 0.9],},
                    { 'value': 4.0, 'gamma': 0.5, 'milestone_fracs': [0.5, 9.0],},
                ],
            },
        },
    }
}
