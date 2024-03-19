import argparse
import logging
import os
from collections import namedtuple
import time
from functools import partial
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms
from tqdm import trange, tqdm
from ddist.data import get_dataset
from ddist.data.preprocessors import _TorchvisionTransforms
from ddistexps.baseline.expcfg import get_candgen
from ddistexps.ftdistil.dataflow2 import CF10CTransforms, AugDataFlow
from ddistexps.teachers import ClipCIFAR10_args
from torchvision import transforms
import wandb



class Config:
    BATCH_SIZE: int = 32
    EPOCHS: int = 2
    OPTIMIZER: str = "SGD"
    LEARNING_RATE: float = 0.1
    WEIGHT_DECAY: float = 5e-4
    MOMENTUM: float = 0.9
    LEARNING_RATE_SCHEDULER: str = "MultiStepLR"
    LEARNING_RATE_SCHEDULER_FRACS: list[float] = [0.3, 0.6, 0.9]
    LEARNING_RATE_GAMMA: float = 0.2
    LOSS_TEMPERATURE_VALUE: float = 1.0
    LOSS_TEMPERATURE_GAMMA: float = 1.0
    LOSS_TEMPERATURE_VALUE_FRACS: list[float] = [0.5, 0.9]
    LOSS_DISTILL_REG: float = 1.0
    LOSS_XENTROPY_REG: float = 1.0
    MODEL_NAME: str = "resnetsmall"  # ['resnetsmall', 'resnetlarge', 'resnetdebug']
    DATASET_NAME: str = "CIFAR10"  # ['CIFAR10', 'TinyCIFAR10', 'CIFAR100']
    STUDENT_CANDIDATE_NUM: int = 3  # [0,1,2,3,4,5]
    CHECKPOINTS_PATH: str = './checkpoints'

    @staticmethod
    def populate_args(args):
        Config.STUDENT_CANDIDATE_NUM = args.candidate_number
        Config.EPOCHS = args.epochs
        Config.MODEL_NAME = args.resnet_subtype

        if not os.path.exists(Config.CHECKPOINTS_PATH):
            os.mkdir(Config.CHECKPOINTS_PATH)

    @staticmethod
    def populate_sweep_config(sweep_config):
        Config.LOSS_TEMPERATURE_VALUE = sweep_config.loss_temperature_cfg['value']
        Config.LOSS_TEMPERATURE_GAMMA = sweep_config.loss_temperature_cfg['gamma']
        Config.LOSS_TEMPERATURE_VALUE_FRACS = sweep_config.loss_temperature_cfg['milestone_fracs']

        Config.LEARNING_RATE_SCHEDULER_FRACS = sweep_config.lr_scheduler_milestone_fracs
        Config.WEIGHT_DECAY = sweep_config.lr_weight_decay


NUM_CANDIDATES = {'resnetsmall': 6, 'resnetlarge': 12, 'resnetdebug': 3}


class LOG:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def __call__(self, *args, **kwargs):
        if args:
            logging.info(*args)
        if kwargs:
            logging.info(kwargs)
            if 'epoch_summary' in kwargs.keys():
                wandb.log(kwargs['epoch_summary']._asdict())
            else:
                wandb.log(kwargs)


logger = LOG()
EpochSummary = namedtuple('EpochSummary',
                          ['epoch', 'loss', 'distill_loss', 'xentropy_loss', 'ep_temperature', 'val_acc'])


class DatasetFactory:
    DATASETS_HUB = {'CIFAR10': CIFAR10, 'CIFAR100': CIFAR100}
    DATASETS_DIR = f"{str(Path.home())}/datasets/"
    NORMALIZATIONS = {'CIFAR10': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      'CIFAR100': transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}
    TRANSFORMS_HUB = {'CIFAR10': CF10CTransforms}
    # add shifts:
    TRANSFORMS_SHIFTS = {'CIFAR10': AugDataFlow}

    def __init__(self, dataset_name):
        assert dataset_name in DatasetFactory.DATASETS_HUB, (f'Expected dataset name one of'
                                                             f' {DatasetFactory.DATASETS_HUB.keys()}.'
                                                             f' Got {dataset_name}')

        torchvision_transforms: _TorchvisionTransforms = DatasetFactory.TRANSFORMS_HUB[dataset_name]()
        torchvision_transforms1: _TorchvisionTransforms = DatasetFactory.TRANSFORMS_SHIFTS[dataset_name]()
        dataset_ctor = DatasetFactory.DATASETS_HUB[dataset_name]
        dataset_dir = DatasetFactory.DATASETS_DIR + dataset_name
        dataset = dataset_ctor(
            root=dataset_dir,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), torchvision_transforms.get_transform('train')])
        )

        self._test_set = dataset_ctor(
            root=dataset_dir,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), torchvision_transforms1.get_transform('val')])
        )
        val_size = len(self._test_set)  # 10000
        train_size = len(dataset) - val_size

        self._train_set, self._val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    @property
    def train_set(self) -> torch.utils.data.Dataset:
        return self._train_set

    @property
    def val_set(self) -> torch.utils.data.Dataset:
        return self._val_set

    @property
    def test_set(self) -> torch.utils.data.Dataset:
        return self._test_set

def get_optimizer(params) -> optim.Optimizer:
    """
    Create an optimizer according to config and parameters
    :param params: the parameters (trainable weights and biases)  of the model
    :return optimizer: the optimizer created
    """
    assert Config.OPTIMIZER in ["Adam", "SGD"], \
        f"Invalid optimizer type. Expected one of [Adam, SGD]. Got {Config.OPTIMIZER}"

    optimizer: optim.Optimizer = optim.Adam(params, lr=Config.LEARNING_RATE) \
        if Config.OPTIMIZER == "Adam" \
        else (optim.SGD(params, lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY))

    return optimizer

def get_lr_scheduler(optimizer: optim.Optimizer) -> optim.lr_scheduler:
    assert Config.LEARNING_RATE_SCHEDULER in ["CosineAnnealingLR", "MultiStepLR"], \
        (f"Invalid learning rate scheduler type."
         f" Expected one of [CosineAnnealingLR, MultiStepLR]. Got {Config.LEARNING_RATE_SCHEDULER}")

    lr_scheduler: optim.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS) \
        if Config.OPTIMIZER == "CosineAnnealingLR" \
        else optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=[int(f * Config.EPOCHS) for f in
                                                        Config.LEARNING_RATE_SCHEDULER_FRACS],
                                            gamma=Config.LEARNING_RATE_GAMMA)

    return lr_scheduler

def get_tempr(epoch):
    i = 0
    tempr_milestones = [int(v * Config.EPOCHS) for v in Config.LOSS_TEMPERATURE_VALUE_FRACS]
    while (i < len(tempr_milestones)) and (epoch >= tempr_milestones[i]):
        i += 1
    return Config.LOSS_TEMPERATURE_VALUE * (Config.LOSS_TEMPERATURE_GAMMA ** i)

def distillation_loss(predlogits, tgtlogits, tempr):
    soft_tgt = F.softmax(tgtlogits / tempr, dim=-1)

    # soft_pred = F.softmax(predlogits / tempr, dim=-1)
    # approx_loss = -torch.sum(soft_tgt * soft_pred) / soft_pred.size()[0]

    soft_pred = F.log_softmax(predlogits / tempr, dim=-1)
    approx_loss = torch.sum(soft_tgt * (soft_tgt.log() - soft_pred))  / soft_pred.size()[0]
    approx_loss = approx_loss * (tempr ** 2)
    return approx_loss


def get_loss(distill_loss, cross_entropy_loss):
    return distill_loss * Config.LOSS_DISTILL_REG + cross_entropy_loss * Config.LOSS_XENTROPY_REG

def save_checkpoint(round, val_acc, state, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    cur = "ensemble-" + timestr + f"-{round}-val_acc_{val_acc}.pt"
    fname = os.path.join(out_dir, cur)
    # Save the current sate dict. TODO: Add random suffix
    assert not os.path.exists(fname), "Found existing checkpoint!"
    torch.save(state, fname)
    assert os.path.exists(fname)
    logger(f"Checkpoint saved to: {fname}")
    return fname

def load_checkpoint(ptfile, model, map_location):
    assert ptfile.endswith('.pt')
    ckpt_id = int(ptfile.split('-')[-1][:-3])
    assert ptfile.endswith('-%d.pt' % ckpt_id), '-%d.pt' % ckpt_id
    sd = torch.load(ptfile, map_location=map_location)
    model.load_state_dict(sd)
    return model, ckpt_id

def get_teacher():
    trunk = ClipCIFAR10_args()
    logger(f'Teacher Model {trunk}')
    return trunk

def get_student_model(candidate_number: int) -> torch.nn.Module:
    """
    Get student model candidate from config product space
    :param candidate_number: Index of candidate in config grid
    :return: student model
    """
    ds_meta = get_dataset(Config.DATASET_NAME).metadata
    grid = get_candgen(gridgen=Config.MODEL_NAME, ds_meta=ds_meta)
    assert candidate_number < len(grid),\
        f"Expected candidate number < {len(grid)} for {Config.MODEL_NAME}, Got {candidate_number}"
    cand = grid[Config.STUDENT_CANDIDATE_NUM]
    student = cand['fn'](**cand['kwargs'])
    logger(f'Student Model {student}')
    return student

def get_data_loaders() -> tuple[DataLoader, DataLoader]:
    """
    Get data loaders for train and validation splits
    :return: tuple of the data loaders
    """
    ds_factory = DatasetFactory(dataset_name=Config.DATASET_NAME)
    ds_train, ds_val = ds_factory.train_set, ds_factory.val_set
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=Config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def get_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Get models for teacher and student
    """
    student_model = get_student_model(Config.STUDENT_CANDIDATE_NUM)
    teacher_model = get_teacher()
    return student_model, teacher_model

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: DataLoader) -> tuple[int, int]:
    """
    Evaluate the given model using the data loader
    :param model: the model to evaluate
    :param data_loader: the data loader
    :return: the number of correct predictions and the total number of predictions
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        total_s, correct_s = 0, 0
        for idx, (x_batch, y_batch) in enumerate(data_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            if logits.shape[1] > 1:
                _, predicted = logits.max(1)
            else:
                predicted = torch.sigmoid(torch.squeeze(logits))
                predicted = (predicted > 0.5)
            iny = torch.squeeze(y_batch.to(device))
            correct = predicted.eq(iny)
            total_s += logits.size(0)
            correct_s += correct.sum().item()
            del correct, iny, x_batch, y_batch, predicted, logits

    return correct_s, total_s

def train(model: torch.nn.Module, trunk: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    """
    Train the student model on the provided data loader with the aid of a teacher model
    :param model: the trained student model
    :param trunk: the teacher model
    :param train_loader: the data loader for the training
    :param val_loader: the data loader used to evaluate the student model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model, trunk = model.to(device), trunk.to(device)
    optimizer = get_optimizer(params=model.parameters())
    lr_scheduler = get_lr_scheduler(optimizer)

    correct, total = evaluate(model=trunk, data_loader=val_loader)
    trunk_acc = float(correct) / float(total) * 100.0
    logger(trunk_acc=trunk_acc)
    best_acc = 0.0
    for epoch in range(Config.EPOCHS):
        temperature = get_tempr(epoch=epoch)
        tot_loss, tot_d_loss, tot_x_loss, batches = 0.0, 0.0, 0.0, 0
        model.train()
        pbar_batches = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (x_batch, y_batch) in pbar_batches:
            optimizer.zero_grad()
            inX, iny = x_batch.to(device), y_batch.to(device)
            predlogits = model(inX)
            with torch.no_grad():
                tgtlogits = trunk(inX)
            x_loss = F.cross_entropy(predlogits, iny)
            d_loss = distillation_loss(predlogits, tgtlogits, temperature)
            loss = get_loss(distill_loss=d_loss, cross_entropy_loss=x_loss)
            loss.backward()
            optimizer.step()
            batches += 1
            tot_loss += float(loss)
            tot_d_loss += float(d_loss)
            tot_x_loss += float(x_loss)

            pbar_batches.set_postfix({
                'total distillation loss': tot_d_loss / batches,
                'total xentropy loss': tot_x_loss / batches,
                'total loss': tot_loss / batches})
            del x_batch, y_batch, predlogits, tgtlogits, x_loss, d_loss, loss
        lr_scheduler.step()

        correct, total = evaluate(model=model, data_loader=val_loader)
        val_acc = float(correct) / float(total) * 100.0
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(round=epoch, val_acc=val_acc,
                            out_dir=f'{Config.CHECKPOINTS_PATH}/{Config.MODEL_NAME}_student_{Config.STUDENT_CANDIDATE_NUM}',
                            state={'epoch': epoch, 'model_state_dict': model.state_dict(),
                                   'optimizer_state_dict': optimizer.state_dict(), 'val_acc': val_acc})
        epoch_summary = EpochSummary(epoch=epoch,
                                     loss=tot_loss / batches,
                                     distill_loss=tot_d_loss / batches,
                                     xentropy_loss=tot_x_loss / batches,
                                     ep_temperature=temperature,
                                     val_acc=val_acc)
        logger(epoch_summary=epoch_summary)

def single_train(args):
    logger(args)
    Config.populate_args(args)

    student_model, teacher_model = get_models()
    train_loader, val_loader = get_data_loaders()
    train(model=student_model, trunk=teacher_model, train_loader=train_loader, val_loader=val_loader)

def sweep_train(sweep_id, args, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})

        args.candidate_number = config.candidate_number
        args.resnet_subtype = config.resnet_subtype
        if config.candidate_number >= NUM_CANDIDATES[args.resnet_subtype]:
            return

        Config.populate_sweep_config(sweep_config=config)

        wandb.run.name = f'{config.resnet_subtype}_candidate_{args.candidate_number}'

        single_train(args)

def run_sweep(args):
    sweep_config = {
        'method': 'grid'
    }
    parameters_dict = {
        'candidate_number': {
            'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        },
        'resnet_subtype': {
            'values': ['resnetsmall', 'resnetlarge']#, 'resnetdebug']
            #'values': ['resnetdebug']
        },
        'loss_temperature_cfg': {
            'values': [
                {'value': 1.0, 'gamma': 1.0, 'milestone_fracs': [1.0]},
                {'value': .3, 'gamma': 1.0, 'milestone_fracs': [0.5, 0.9]},
                {'value': 4.0, 'gamma': 0.5, 'milestone_fracs': [0.5, .9]}
            ],
        },
        'lr_scheduler_milestone_fracs': {
            'values': [[.5, 0.75], [.3, 0.6, 0.9]]
        },
        'lr_weight_decay': {
            'values': [5e-4, 5e-3]
        }
    }
    sweep_config['parameters'] = parameters_dict
    metric = {
        'name': 'val_acc',
        'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    sweep_id = wandb.sweep(sweep_config, project="CMU-DISTILLATION")

    wandb.agent(sweep_id, partial(sweep_train, sweep_id=sweep_id, args=args))

if __name__ == '__main__':
    # transform_names = ['brightness','contrast','gaussian_blur','pixelate','saturate','gaussian_noise','impulse_noise']
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--resnet-subtype', type=str, default=['resnetsmall', 'resnetlarge'],
                        choices=['resnetsmall', 'resnetlarge', 'resnetdebug'])
    parser.add_argument('--candidate-number',
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 3, 4, 5],
                        help='number of candidate grid entry')
    parser.add_argument('--sweep', action='store_true',default=True, help='run a sweep. otherwise run single')
    #parser.add_argument('--corruption', default = [help='run a sweep of fine tuning')
    args = parser.parse_args()
    if args.sweep:
        logger('Starting sweep')
        api_key = 'de945abee07d10bd254a97ed0c746a9f80a818e5'
        wandb.login(key=api_key)
        run_sweep(args)
    else:
        logger('Starting a single train')
        api_key = 'de945abee07d10bd254a97ed0c746a9f80a818e5'
        wandb.login(key=api_key)
        wandb.init(project="CMU-DISTILLATION", entity="ariel_solomon", name=f'{args.resnet_subtype}')
        single_train(args)
        # EASY_C_SHIFTS = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur']
        # MEDIUM_C_SHIFTS = ['jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
        # HARD_C_SHIFTS = ['gaussian_noise', 'glass_blur', 'impulse_noise', 'shot_noise']
        # INET_C_SHIFTS = ['brightness', 'contrast', 'elastic_transform', 'fog',
        #                 'frost', 'jpeg_compression', 'pixelate', 'snow']
        # import torch
        # from torchvision import transforms
        # from torchvision.datasets import CIFAR10
        # import random  # For random transform selection and parameter adjustments
        #
        # def transform_cifar10(data_dir, transform_list, download=True):
        #
        #     transform = []
        #     for transform_name in transform_list:
        #         if transform_name == 'brightness':
        #             transform.transforms.append(transforms.ColorJitter(brightness=random.uniform(0.5, 1.5)))
        #         elif transform_name == 'contrast':
        #             transform.transforms.append(transforms.ColorJitter(contrast=random.uniform(0.5, 1.5)))
        #         elif transform_name == 'gaussian_blur':
        #             transform.transforms.append(transforms.GaussianBlur(kernel_size=random.randint(3, 7), sigma=random.uniform(0.1, 2.0)))
        #         elif transform_name == 'pixelate':
        #             transform.transforms.append(transforms.Pixelate(pixel_size=random.randint(2, 8)))
        #         elif transform_name == 'saturate':
        #             transform.transforms.append(transforms.ColorJitter(saturation=random.uniform(0.5, 1.5)))
        #         elif transform_name == 'gaussian_noise':
        #             transform.transforms.append(transforms.GaussianNoise(mean=0.0, std=random.uniform(0.01, 0.1)))
        #         elif transform_name == 'impulse_noise':
        #             # Custom implementation (example)
        #             def impulse_noise(img):
        #                 noise_prop = random.uniform(0.01, 0.05)  # Proportion of pixels to modify
        #                 return transforms.functional.random_noise(img, mode='pepper', p=noise_prop)
        #             transform.transforms.append(impulse_noise)
        #         else:
        #             raise ValueError(f"Unsupported transformation: {transform_name}")
        #
        #     # Dataset loading
        #     dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=download)
        #
        #     return dataset
