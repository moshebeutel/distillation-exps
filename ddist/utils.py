import inspect
import os
import time
from math import floor
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
import types
import itertools
import pandas as pd
from argparse import Namespace
import itertools
import pandas as pd
import mlflow
import ray
from collections.abc import MutableMapping


def extract_image_patches1(images, kh, kw, sh=1, sw=1, pad_h=0, pad_w=0):
    '''
    images: Image tensor to extract patches from. We assume the
        image follows [batch_size, channels, H, W].

    kh, kw: Height of patch and width of patch
    sh, sw: Stride on height and width between patches.
    pad_h, pad_w: Padding on width and height.
    '''
    lg = CLog
    msg = "Invalid shape. We expect [batch, channels, H, W]"
    assert len(images.shape) == 4, msg
    # F.Pad takes a pair of arguments for each dimension we
    # want to pad (as a list). Padding starts from the last
    # dimension. Here we have (top, bottom, left, right)
    # Before: [batch, channels, H, W]
    # After: [batch, channels, H+2*pad_h, W + 2*pad_w]
    lg.debug('Shape before padding: ', images.shape)
    x = F.pad(images, (pad_h, pad_h, pad_w, pad_w))
    lg.debug('Shape after padding: ', x.shape)
    # Get patches of size (kh, W) along the height dimension.
    # Assuming no padding;
    # Before: [batch, channels, H, W]
    # After: [batch, channels, num_patches, W, kh]
    # That is, we get patches of size (Wxkh)
    patches = x.unfold(2, kh, sh)
    lg.debug('Shape after unfolding along H: ',
             patches.shape, "kh, sh: ", kh, sh)
    # Before: [batch, channels, num_patches, W, kh]
    # After:  [batch, channels, num_patch_h, num_patch_w, kh, kw]
    patches = patches.unfold(3, kw, sw)
    lg.debug('Shape after unfolding along W: ',
             patches.shape, "kw, sw: ", kw, sw)
    # Before:  [batch, channels, num_patch_h, num_patch_w, kh, kw]
    # After:  [batch,  num_patch_h, num_patch_w, kh, kw, channels]
    # Reorder to [batch, num_patch_h, num_patch_w, kh, kw, channels]
    patches = patches.permute((0, 2, 3, 4, 5, 1))
    lg.debug('Shape after permuting: ', patches.shape)
    return patches


class CLog:
    '''
    Simple colored logger.

    Note: Do not implement pprint as part of this. It is hard to handle. We
    expect user to use pprint.pforamat to obtain a formatted string and provide
    that as the argument to functions here.

    We pick up logging states from environmental variables

    Call init() from colorama to enable colors.
    '''

    CLOG_COLOR_ON = False
    if 'CLOG_COLOR_ON' in os.environ:
        CLOG_COLOR_ON = bool(int(os.environ['CLOG_COLOR_ON']))
    # initialize colorama
    # cinit(strip=False)
    console = Console(width=200)

    @staticmethod
    def cpt(pre, *args, **kwargs):
        callerframerecord = inspect.stack()[2]
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        s = '%s:%s:' % (info.function, info.lineno)
        echo = CLog.console.print
        if CLog.CLOG_COLOR_ON:
            # echo(pre, s, *args, Style.RESET_ALL, **kwargs, no_wrap=True)
            echo(pre, s, *args, **kwargs)
        else:
            echo(s, *args, **kwargs)

    @staticmethod
    def debug(*args, **kwargs):
        # Regular print
        s = '[red][Debug] '
        CLog.cpt(s, *args, **kwargs)

    @staticmethod
    def info(*args, **kwargs):
        s = '[blue][Info ]'
        CLog.cpt(s, *args, **kwargs)

    @staticmethod
    def warn(*args, **kwargs):
        CLog.warning(*args, **kwargs)

    @staticmethod
    def warning(*args, **kwargs):
        # Regular print
        # s = Fore.YELLOW + '[Warn ] '
        s = '[yellow][Warn ] '
        CLog.cpt(s, *args, **kwargs)

    @staticmethod
    def fail(*args, **kwargs):
        callerframerecord = inspect.stack()[1]
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        fn = os.path.basename(info.filename)
        # Regular print
        # s = Fore.RED + '[Fail ] %s:' % (fn)
        s = '[red][Fail ] %s:' % (fn)
        CLog.cpt(s, *args, **kwargs)

    @staticmethod
    def iinfo(*args, **kwargs):
        s = '[background white blue][Info ]'
        # s = Back.WHITE + Fore.BLUE
        # s += '[INFO] '
        CLog.cpt(s, *args, **kwargs)

    @staticmethod
    def print_df(df, title='info'):
        # from rich_dataframe import prettify
        # table = prettify(df)
        table = Table(title)
        flt_fmt = lambda x: '%.4f' % x
        s = df.to_string(float_format=flt_fmt)
        table.add_row(s)
        CLog.cpt('', table)


NullLogger = None
def get_logger(verbose):
    if verbose is True: return CLog
    # We will modify the global to prevent duplication
    global NullLogger
    if NullLogger is not None: return NullLogger
    def noop(*args, **kwargs): pass
    LOG_FUNC_LIST = ['debug', 'info', 'warn', 'fail', 'iinfo']
    attr_kwargs = {k: noop for k in LOG_FUNC_LIST}
    lg_noop = types.SimpleNamespace(**attr_kwargs)
    # global NullLogger
    NullLogger = lg_noop
    return NullLogger


def dataset_with_indices(cls):
    """
    WARNING: cls is a class not an instance.

    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.

    From: https://discuss.pytorch.org/7948/19
    Quite a cute bit of python
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)

    Kind gentelman here:
    https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    num = (h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1)
    h = floor((num / stride) + 1)
    num = (h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1)
    w = floor((num / stride) + 1)
    return h, w


def get_exp_scale(num_changes, exp=1):
    """
    Produce a biased schedule. Higher values of exp, biases the sequence to
    spend more epochs between early switches. Smaller values of exp has the
    opposite effect

    num_changes=6, exp=1: [0.  , 0.36, 0.64, 0.84, 0.96, 1.  ]
    """
    a = np.linspace(0, 1, num_changes)
    a = (1 - ((a**2) ** exp))
    return np.flip(a)


def indicator_image(indicators, ref_index):
    """
    Construct a 2d image from an indicator vector,
    using the index ordering specified in ref_index.
        
    The indicator image is filled from the first index and iterating onward. If
    at some point few elements than desired are found, the final pixels will be
    left unfilled.
    """
    horizontal = int(np.sqrt(len(indicators)))
    vertical = int(np.ceil(len(indicators) / horizontal))
    assert horizontal * vertical >= len(indicators)
    image_base = np.zeros((vertical, horizontal))
    for position, elem_idx in enumerate(ref_index):
        i, j = int(position / horizontal), position % horizontal
        image_base[i, j] = indicators[elem_idx]
    return image_base


def get_reference_indexing(base_sort, row_sort):
    """
    base_sort: A array containing the first axis of sort.
        The indices will be sorted on base_sort *before* converting
        then to a image (2D).
    row_sort: Once the image is constructed, we use this to sort
        each row within itself.
    
    We first sort by base_sort and produce the first set of indices. This is a
    reverse sort so that large values move towards the top. We then row sort
    based on the values in row_sort
    
    (Remember there could be in-accuracies in the last row)
    """
    # Sort the main set of elements in base_sort
    _index_list = np.flip(np.argsort(base_sort))
    # Now we use this base sort to place elements in row_sort onto an image. We
    # then compute the row-level (relative to row) sorting indices.
    nonrowsorted = indicator_image(row_sort, _index_list)
    rowsorted = np.argsort(nonrowsorted, axis=1)
    # We now have the row-sorted arguments, but we need to convert them to
    # absolute indexing. We do that by first computing an image of the full
    # indexing, and then picking the row-level values (absolute indices) based
    # on rowsorted (relative) indexing.
    refimg = indicator_image(
        _index_list, np.arange(len(_index_list)).astype(int))
    ref_list = np.take_along_axis(refimg, rowsorted, axis=1)
    ref_list = np.reshape(ref_list, -1).astype(int)[:len(_index_list)]
    # ref_index_list = _index_list
    # ref_col_index_list = ref_list
    return ref_list



class Weighting:
    """
    Methods used to compute sample weights from logk etc.
    """
    @staticmethod
    def get_sample_weights(logk):
        """
        For each data point, we compute \abs(Kp - Kn) and sum it up along the
        features direction. Since the contribution of a data point into the
        final inner product is proportional to Kp-Kn and on Kp+Kn, this seems
        like the better weighting scheme.
        """
        # 1: Covered
        d = int(logk.shape[1]/2)
        kp, kn = logk[:, :d], logk[:, d:]
        assert np.abs(np.sum(np.exp(logk)) - d) < 1e-4
        w = np.abs(np.exp(kp) - np.exp(kn))
        sweights = np.sum(w, axis=1)
        if np.sum(sweights) > 0:
            sweights = sweights / np.sum(sweights)
        else:
            # Sum is exact zero only when distributions are uniform
            sweights[:] = 1.0/  len(sweights)
        assert sweights.ndim == 1, sweights.shape
        assert np.abs(np.sum(sweights)- 1) <= 1e-5, np.sum(sweights)
        assert len(sweights) == len(logk)
        return sweights

    @staticmethod
    def almost_equal_partitions(sample_weights, num_partitions):
        """Split the sample_weights into num_partitions so that each partition
        supports (approximately) equal probability mass.

        Returns end-points of bounds (exclusive)"""
        # We have sample weights
        shardmass = (1.0 / num_partitions)
        start, end, bounds = 0, 1, []
        rank, currsum, cumsum = 0, 0, np.cumsum(sample_weights)
        cumsum = cumsum / cumsum[-1]
        for end in range(1, len(cumsum)+1):
            currsum = cumsum[end-1]
            if currsum < (rank + 1) * shardmass:
                continue
            # We have reached the end of this shard
            bounds.append((start, end))
            start = end
            rank += 1
        return bounds

    @staticmethod
    def get_num_samples_for(sweights, mass=0.95):
        """Number of data points that contain mass"""
        assert 0 <= mass <= 1, mass
        # Reverse sort
        wght = -1 * np.sort(-sweights)
        for i in range(len(wght)):
            if np.sum(wght[:i+1]) >= mass:
                break
        assert np.sum(wght[:i+1]) >= mass, np.sum(wght[:i+1])
        return i + 1, np.sum(wght[:i+1])


def product_of_dict_of_lists(**kwargs):
    """Returns the product space of dict of lists. Non-recursive.
        {k1: [v1, v2, v3]} --> [{k1:v1}, {k1:v2}, {k1:v3}]
    """
    keys = kwargs.keys()
    itr = []
    for instance in itertools.product(*kwargs.values()):
        itr.append(dict(zip(keys, instance)))
    return itr


def spec_to_prodspace(verbose_depth=1, **kwargs):
    """Recursively convert dict of lists to records
    Example:
    {k1: [v1, v2, v3]} --> [{k1:v1}, {k1:v2}, {k1:v3}]

    {
        k1: [v1, v2, v3]
        k2: {k21: [v21, v22], k22: [c22]}
    } 
    --> 
    {
        k1: [v1, v2, v3],
        k2: [{k21:v21, k22: c22},  {k21: v22, k22: c22}]
    }
    -->
    [
        {k1:v1, k2: {k21:v21, k22: c22}}
        {k1:v1, k2: {k21:v22, k22: c22}}
        {k1:v2, k2: {k21:v21, k22: c22}}
        {k1:v2, k2: {k21:v22, k22: c22}}
        {k1:v3, k2: {k21:v21, k22: c22}}
        {k1:v3, k2: {k21:v22, k22: c22}}
    ]
    Algorithm:
        for key, val in dict:
            if val is list: continue
            val = dict_to_list(val)
        # All vals are lists. Multiply
        return product_dict(**)
    """
    for key in kwargs.keys():
        val = kwargs[key]
        if type(val) not in [list, dict]:
            kwargs[key] = [val]

        val = kwargs[key]
        if type(val) == list:
            continue
        new_val = spec_to_prodspace(verbose_depth=verbose_depth-1, **val)
        kwargs[key] = new_val

    cfg_names = list(kwargs.keys())
    cfg_counts = [len(kwargs[key]) for key in cfg_names]
    lg = CLog
    if verbose_depth > 0:
        _df = pd.DataFrame({'name': cfg_names, 'count': cfg_counts})
        lg.print_df(_df, "Product space coordinate sizes.")
    prod_list = product_of_dict_of_lists(**kwargs)
    if verbose_depth > 0:
        lg.info("Product space size: ", len(prod_list))
    return prod_list


def flatten_dict(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def namespace_to_dict(args, depth_limit=6, exclude=['module', 'trunk']):
    """Recursively convert namespace object to dict object"""
    if depth_limit == 0:
        return args
    args1 = args
    if type(args) not in [dict]:
        args1 = dict(vars(args))
    args = {k:v for k, v in args1.items() if k not in exclude}
    for key in args.keys():
        if key in exclude: continue
        if isinstance(args[key], Namespace):
            args[key] = namespace_to_dict(args[key], depth_limit-1)
    return args 

def dict_to_namespace(args, depth_limit=6):
    """Recursively convert dict object to namespace"""
    if depth_limit == 0:
        return args
    for key in args.keys():
        if isinstance(args[key], dict):
            args[key] = dict_to_namespace(args[key], depth_limit-1)
    return Namespace(**args)



def save_checkpoint(round, wlgen, out_dir, remove_previous=False):
    state = {}
    state['wlgen'] = ray.get(wlgen.state_dict.remote())
    prv = "ensemble-%d.pt" % (round - 1)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    prev_fname = os.path.join(out_dir, prv)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cur = "ensemble-" + timestr + f"-{round}.pt"
    fname = os.path.join(out_dir, cur)
    # Save the current sate dict. TODO: Add random suffix
    assert not os.path.exists(fname), "Found existing checkpoint!"
    torch.save(state, fname)
    assert os.path.exists(fname)
    # Remove previous
    if remove_previous:
        CLog.info("Removing: ", prev_fname)
        os.remove(prev_fname)
    CLog.info("Checkpoint saved to: ", fname)
    return fname


def load_checkpoint(ptfile, wlgen, map_location):
    assert ptfile.endswith('.pt')
    ckpt_id = int(ptfile.split('-')[-1][:-3])
    assert ptfile.endswith('-%d.pt' % ckpt_id), '-%d.pt' % ckpt_id
    sd = torch.load(ptfile, map_location=map_location)
    ray.get(wlgen.load_state_dict.remote(sd['wlgen']))
    return sd, ckpt_id
