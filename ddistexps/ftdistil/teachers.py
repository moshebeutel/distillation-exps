import torch
import argparse
import os
import csv
import torch.nn as nn

import ddist.models.pretrained as pretrained
from ddist.models import RNNClassifier 
from ddist.models.resnet_cifar import resnet20 as resnet20cf10
from ddist.models.resnet_cifar import resnet56 as resnet56cf10
from ddist.models.clip import clip_vitb32
from torchvision.models import resnet50 as resnet50imgnet
from torchvision.models import resnet101 as resnet101imgnet
from ddist.models.densenet_cifar import densenet121, densenet169
from ddist.models.resnet_cifar import resnet56 as resnet56tin
from ddist.models.resnet_cifar import resnet110 as resnet110tin
from ddist.utils import CLog as lg

class NoOpTeacher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[:, 0, 0, 0]
        frac, flr = (x/2), torch.floor(x/2)
        even = (torch.abs(frac - flr) < 0.25) * 1.0
        odd = (torch.abs(frac - flr) > 0.25) * -1.0
        even = even.to(dtype=torch.float32)
        odd = odd.to(dtype=torch.float32)
        return torch.unsqueeze(even + odd, dim=-1)


class Trunks:
    """
    Add all trunk initilization code here.
    """

    @staticmethod
    def __call__(trunk_cfg):
        name = trunk_cfg.name
        trunk = None
        if 'CIFAR10ResNet' in name:
            trunk = Trunks.ResNetCIFAR10(trunk_cfg)
        elif 'RangeTrunk' in name:
            trunk = Trunks.RangeTrunk(trunk_cfg)
        elif 'DenseNet' in name:
            trunk = Trunks.DenseNetCIFAR100(trunk_cfg)
        elif 'ClipCIFAR10' in name:
            trunk = Trunks.ClipCIFAR10_args(trunk_cfg)
        else:
            raise ValueError("Unknown trunk:", name)
        return trunk

    @staticmethod
    def RangeTrunk(args):
        """Custom trunk for range dataset. Note that the input shape of trunk is
        controlled from RangeDataset."""
        return NoOpTeacher()

    @staticmethod
    def ResNetCIFAR10(trunk_cfg):
        """ Returns trunks and sets the relevant arguments for
        ResNet20/ResNet56 on CIFAR10.
        """
        if '20' in trunk_cfg.name:
            sdf = pretrained.RESNET20_CIFAR10
            func = resnet20cf10
        elif '56' in trunk_cfg.name:
            sdf = pretrained.RESNET56_CIFAR10
            func = resnet56cf10
        else: 
            raise ValueError("Unknown teacher:", trunk_cfg.name)

        sd = torch.load(sdf, map_location='cpu')['state_dict']
        sd2 = {}
        for key in sd.keys():
            val = sd[key]
            if key.startswith('module.'):
                key = key[len('module.'):]
            sd2[key] = val
        trunk = func()
        trunk.load_state_dict(sd2)
        return trunk

    @staticmethod
    def DenseNetCIFAR100(args):
        """
        Returns trunks and sets the relevant arguments for DenseNet on CIFAR100
        """
        # These are all bottleneck models
        if '121' in args.trunk_cfg.name:
            sdf = pretrained.DENSENET121_CIFAR100
            func = densenet121
            lg.info(f"Expected trunk test accuracy: 73.08")
        elif '169' in args.trunk_cfg.name:
            sdf = pretrained.DENSENET169_CIFAR100
            func = densenet169
            lg.info(f"Expected trunk test accuracy: 77.99")
        else: 
            raise ValueError("Unknown trunk:", args.trunk_cfg.name)

        sd = torch.load(sdf, map_location='cpu')
        trunk = func()
        trunk.load_state_dict(sd)
        return trunk

    @staticmethod
    def ClipCIFAR10_args(args):
        """Returns trunks and sets the model type. """
        from ddist.data.dataset import CIFAR10Dataset
        labels = CIFAR10Dataset.metadata['labels']

        kwargs = {'labels': labels, 'model_pretrain_save': 'laion2b_s34b_b79k'}
        trunk = clip_vitb32(**kwargs)
        return trunk


# class CLIArgs:
#     SUPP_VISION_DATASETS = ['CIFAR10', 'TinyCIFAR10', 'CIFAR100', 'ImageNet1k',
#                             'TinyImageNet']
#     SUPP_SPEECH_DATASETS = ['GoogleSpeech']
#     CIFAR10_CLIP_TR = ['CIFAR10-ClipViTB32']
#     CIFAR10_TR = ['CIFAR10-ResNet20', 'CIFAR10-ResNet56']
#     CIFAR100_TR = ['CIFAR100-DenseNet121', 'CIFAR100-DenseNet169']
#     IMAGENET_TR = ['ImageNet1k-ResNet101', 'ImageNet1k-ResNet50']
#     TINY_IMAGENET_TR = ['TinyImageNet-ResNet110', 'TinyImageNet-ResNet56']
#     LSTM_TRUNKS = ['GoogleSpeech-64', 'GoogleSpeech-128']
#     SUPP_TRUNKS = CIFAR10_CLIP_TR + CIFAR10_TR + IMAGENET_TR + CIFAR100_TR + TINY_IMAGENET_TR
#     SUPP_DATASETS = SUPP_VISION_DATASETS + SUPP_SPEECH_DATASETS
#     TORCH_DATA_DIR = os.environ['TORCH_DATA_DIR']
#     TRUNK_INFO = {
#         'GoogleSpeech-64': {
#             'mclass': 
#             (RNNClassifier, {'num_ts': 99, 'num_feats':32, 
#                              'out_dim': 13, 'hid_dim_list': [64],
#                              'celltype': 'LSTM'}),
#             'ckpt_file': 'GoogleSpeechB1/GoogleSpeechB1_job_2/model-199.pt',
#             'in_dim': (99, 32), 'out_dim': 13, 'num_samples': 51088
#         },
#         'GoogleSpeech-128': {
#             'mclass': 
#             (RNNClassifier, {'num_ts': 99, 'num_feats':32, 
#                              'out_dim': 13, 'hid_dim_list': [128],
#                              'celltype': 'LSTM'}),
#             'ckpt_file': 'GoogleSpeechB1/GoogleSpeechB1_job_3/model-199.pt',
#             'in_dim': (99, 32), 'out_dim': 13, 'num_samples': 51088
#         },
#     }
#
#     def __init__(self):
#         parser = argparse.ArgumentParser()
#         msg = "Dataset to use for model. Available datasets are: "
#         msg += str(CLIArgs.SUPP_DATASETS)
#         msg2 = "The trunk to use: "
#         msg2 += str(CLIArgs.SUPP_TRUNKS)
#         ddir = CLIArgs.TORCH_DATA_DIR + "/DATASET/"
#         # Exp Details
#         parser.add_argument("--data-set", type=str, help=msg,
#                             required=True, default=None)
#         parser.add_argument("--model-variant", type=str, help=msg2,
#                             required=True)
#         parser.add_argument("--ckpt-file", type=str, default=None,
#                             help="Round level checkpoint file")
#         parser.add_argument("--spec-name", type=str, default=None,
#                             help='Specification-dict name for ensemble.',
#                             required=True)
#         parser.add_argument("--num-ftrl-rounds", type=int, help="FTRL Rounds",
#                             default=6)
#         # General?Config
#         parser.add_argument("--out-dir", type=str, required=False,
#                             help="Directory to save trained models, logs etc")
#         parser.add_argument("--download", default=False,  action='store_true',  
#                             help="Attempt to download dataset.")
#         parser.add_argument("--download-dir", default=None, type=str, 
#                             help="Directory to download datasets to.")
#         # Actor resource config
#         parser.add_argument('--ddp-actor-gpus', type=int, default=0.5,
#                             help="Number of gpus to be allocated to each DDP"
#                             + " actor for training.")
#         parser.add_argument('--measurement-sample-size', default=20, type=int,
#                             help="Number of configs to sample per measurement.")
#         parser.add_argument('--ddp-total-workers', default=8, type=int,
#                             help="Total available GPUs (one worker per gpu)")
#         parser.add_argument('--ddp-num-train-workers', default=4, type=int,
#                             help="Number of workers per job for training.")
#         # On disk files for heavy data (imageNet)
#         # parser.add_argument('--temp-logk', type=str, default=None)
#         # parser.add_argument('--temp-dirtn', type=str, default=None)
#         self.parser = parser
#
#     def csv_to_type(self, str_s, type_def):
#         str_s = str_s.strip()
#         result = list(csv.reader([str_s]))
#         ret_l = []
#         if len(result) > 0:
#             assert len(result) == 1
#             result = result[0]
#             for elem in result:
#                 ret_l.append(type_def(elem))
#         return ret_l
#
#     def check_paths(self, args):
#         # Checkpoint parent directory
#         assert os.path.exists(args.out_dir), "Out director does not exist"
#         assert os.path.isdir(args.out_dir), f'Not a directory: {args.out_dir}'
#         # Data download directory
#         if args.download_dir is None:
#             pth = os.path.join(CLIArgs.TORCH_DATA_DIR, args.data_set)
#             args.download_dir = pth
#         if not os.path.exists(args.download_dir):
#             os.mkdir(args.download_dir)
#         files = os.listdir(args.out_dir)
#         ifiles = [f for f in files if f.endswith('.pt')]
#         if len(ifiles) > 0:
#             lg.warning("Checkpoint files found: \n\t" + str(ifiles))
#             lg.warning("In dir: ", args.out_dir)
#         if len(files) > 0:
#             lg.warning("Save directory not-empty:\n\t" + str(files))
#             lg.warning("In dir: ", args.out_dir)
#
#
#     def ResNetCIFAR10_args(self, args):
#         """ Returns trunks and sets the relevant arguments for
#         ResNet20/ResNet56 on CIFAR10.
#         """
#
#         if '20' in args.model_variant:
#             sdf = pretrained.RESNET20_CIFAR10
#             func = resnet20cf10
#
#         elif '56' in args.model_variant:
#             sdf = pretrained.RESNET56_CIFAR10
#             func = resnet56cf10
#
#         args.trunk_out_dim = 10
#         from ddist.data.dataset import (
#             CIFAR10Dataset, TinyCIFAR10Dataset, CIFAR100Dataset
#         )
#         if args.data_set in ['CIFAR10']:
#             meta = CIFAR10Dataset.metadata
#         elif args.data_set in ['CIFAR100']:
#             meta = CIFAR100Dataset.metadata
#         else: 
#             meta = TinyCIFAR10Dataset.metadata
#         args.num_train_samples = meta['num_train_samples']
#         args.model_type = 'net'
#
#         sd = torch.load(sdf, map_location='cpu')
#         prec, sd = sd['best_prec1'], sd['state_dict']
#         sd2 = {}
#         for key in sd.keys():
#             val = sd[key]
#             if key.startswith('module.'):
#                 key = key[len('module.'):]
#             sd2[key] = val
#         trunk = func()
#         trunk.load_state_dict(sd2)
#         lg.info("Reported trunk (teacher) test accuracy: ", prec)
#         return args, trunk

    # def ResNetImageNet_args(self, args):
    #     """ Returns trunks and sets the relevant arguments for
    #     ResNet50/ResNet101 on ImageNet
    #     """
    #     if '50' in args.model_variant:
    #         trunk = TrunkWrapper(resnet50imgnet, weights="DEFAULT")
    #         exp_acc = 76.13
    #     elif '101' in args.model_variant:
    #         trunk = TrunkWrapper(resnet101imgnet, weights="DEFAULT")
    #         exp_acc = 77.17
    #
    #     args.trunk_out_dim = 1000
    #     args.num_train_samples = 1281167
    #     args.model_type = 'net'
    #     lg.info("Expected trunk test accuracy: ", exp_acc)
    #     #https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html
    #     return args, trunk
    #
    # def ResNetTinyImageNet_args(self, args):
    #     """ Returns trunks and sets the relevant arguments for
    #     ResNet56/ResNet110 on TinyImageNet
    #     """
    #     if '56' in args.model_variant:
    #         raise ValueError(f"We dont have this weights {resnet56in.__name__()}")
    #         sdf = pretrained.RESNET50_TINYIMAGENET
    #         trunk = TrunkWrapper(resnet56tin, num_classes=200)
    #     elif '110' in args.model_variant:
    #         sdf = pretrained.RESNET110_TINYIMAGENET
    #         trunk = TrunkWrapper(resnet110tin, num_classes=200, sdf=sdf)
    #
    #     args.trunk_out_dim = 200
    #     args.num_train_samples = 200 * 500
    #     args.model_type = 'net'
    #     # lg.info("Expected trunk test accuracy: ", 54.5)
    #     return args, trunk
    #
    #
    # def LSTM_args(self, args):
    #     """Return trunks based on LSTM arguments. 
    #     """
    #     tinfo = CLIArgs.TRUNK_INFO[args.model_variant]
    #     mclass, margs = tinfo['mclass']
    #     ckpt = os.environ['DDIST_EXPS_DIR']
    #     ckpt = os.path.join(ckpt, tinfo['ckpt_file'])
    #     sd = torch.load(ckpt)
    #     trunk = mclass(**margs)
    #     trunk.load_state_dict(sd)
    #     trunk.eval()
    #     trunk.requires_grad = False
    #     for n, p in trunk.named_parameters():
    #         p.requires_grad = False
    #     args.in_dim = tinfo['in_dim']
    #     args.trunk_out_dim = tinfo['out_dim']
    #     args.num_train_samples = tinfo['num_samples']
    #     args.model_type = 'lstm'
    #     return args, trunk


    # def make_args(self, commands=None):
    #     args = self.parser.parse_args(args=commands)
    #     msg = f"Unsupported datset. We only support {CLIArgs.SUPP_DATASETS}"
    #     assert args.data_set in CLIArgs.SUPP_DATASETS, msg
    #     self.check_paths(args)
    #     assert args.model_variant in CLIArgs.SUPP_TRUNKS, CLIArgs.SUPP_TRUNKS
    #     if args.model_variant in CLIArgs.CIFAR10_CLIP_TR:
    #         args, trunk = self.ClipCIFAR10_args(args)
    #     elif args.model_variant in CLIArgs.CIFAR10_TR:
    #         args, trunk = self.ResNetCIFAR10_args(args)
    #     elif args.model_variant in CLIArgs.CIFAR100_TR:
    #         args, trunk = self.DenseNetCIFAR100_args(args)
    #     elif args.model_variant in CLIArgs.IMAGENET_TR:
    #         args, trunk = self.ResNetImageNet_args(args)
    #     elif args.model_variant in CLIArgs.TINY_IMAGENET_TR:
    #         args, trunk = self.ResNetTinyImageNet_args(args)
    #     elif args.model_variant in CLIArgs.LSTM_TRUNKS:
    #         args, trunk = self.LSTM_args(args)
    #     else:
    #         lg.fail("Unsupported model: ", args.model_variant)
    #         exit()
    #     # Add resource requirements
    #     args.ddp_resource_req = {
    #         # 'num_cpus': args.ddp_cpus_per_gpu,
    #         'num_gpus': args.ddp_actor_gpus,
    #     }
    #     return args, trunk
