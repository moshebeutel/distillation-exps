import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ddist.utils import CLog as lg


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x): return x

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    

class DenseNN(nn.Module):
    def __init__(self, in_dim, out_dim, fc_dim_list, non_lin='Sigmoid'):
        '''
        Input: [-1, in_dim]
        Output: [-1, out_dim]

        !NOTE! The last layer is linear of the appropriate size. Thus even for
        empty fc_dim_list, we will have a linear layer matching input output
        shapes.
        '''
        super(DenseNN, self).__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.fc_dim_list = fc_dim_list
        self.non_lin = non_lin
        self.fn_nonlin = self.__get_fn_nonlin(non_lin)
        self.hid_layer_list = nn.ModuleList()
        for fc_dim in fc_dim_list:
            layer = nn.Linear(in_dim, fc_dim)
            in_dim = fc_dim
            self.hid_layer_list.append(layer)
        self.linear_layer = nn.Linear(in_dim, out_dim)

    def __get_fn_nonlin(self, non_lin):
        if non_lin == 'ReLU':
            return F.relu
        elif non_lin == 'Sigmoid':
            return torch.sigmoid
        else:
            s = ['ReLU']
            lg.fail("Unsupported non-linear op:", non_lin,
                    f". Supported ops {s}")

    def forward(self, x):
        fn_nonlin = self.fn_nonlin
        out = x
        for layer in self.hid_layer_list:
            out = fn_nonlin(layer(out))
        return self.linear_layer(out)

class RNNClassifier(nn.Module):
    def __init__(self, num_ts, num_feats, hid_dim_list, out_dim,
                 celltype='LSTM', dropout=0.0):
        """
        Simple RNN based classifier. RNN configuration is specified as a list
        of hidden dimensions.
        """
        super(RNNClassifier, self).__init__()
        self.celltype = celltype
        self.num_ts, self.num_feats, self.out_dim = num_ts, num_feats, out_dim
        self.hid_dim_list = hid_dim_list
        cellfn = self.__get_cell(celltype)
        inpd, cell_l = num_feats, []
        self.dropout = nn.Sequential()
        if dropout > 0.0:
            lg.info("Dropout will be used.")
            self.droput = nn.Dropout(p=dropout)
        for i, hidd in enumerate(hid_dim_list):
            cell = cellfn(input_size=inpd, hidden_size=hidd, num_layers=1,
                          batch_first=True)
            inpd = hidd
            cell_l.append(cell)
        self.cell_list = nn.ModuleList(cell_l)
        self.linear = nn.Linear(hid_dim_list[-1], out_dim)

    def __get_cell(self, celltype):
        assert celltype in ['LSTM', 'GRU']
        if celltype == 'LSTM':
            return nn.LSTM
        elif celltype == 'GRU':
            return nn.GRU

    def get_cell_out(self, output):
        '''
        Depending on the RNN celltype used, we need to do a bit of work to get
        the output states from the output of RNN layer. For example, for LSTM
        cell, the outputs are of the form
            (all layer output, (last layer h, last layer c))
        '''
        celltype = self.celltype
        if celltype == 'LSTM':
            # (all layer output, (last layer h, last layer c))
            # Further: last layer h is of shape:
            #   [num_layers, Batch size, hidden dim]
            final_state = output[1][0]
            # There should only be one final-state as we manually unroll cells
            assert len(final_state) == 1
            final_state = torch.squeeze(final_state)
            return final_state
        elif celltype == 'GRU':
            # (all layer output, last_layer_final_stte)
            final_state = output[1]
            # There should only be one final-state as we manually unroll cells
            assert len(final_state) == 1
            final_state = torch.squeeze(final_state)
            return final_state

    def forward(self, x):
        # Initial states default to zero.
        inp = x
        for i, cell in enumerate(self.cell_list):
            inp, ht_ = cell(inp)
            # Don't add dropout for the last layer
            if i == len(self.cell_list) - 1:
                continue
            inp = self.dropout(inp)
        out = self.get_cell_out((inp, ht_))
        out = self.linear(out)
        return out

    def freeze(self):
        for n, p in self.named_parameters():
            p.requires_grad = False

class EarlyRNNClassifier(nn.Module):
    def __init__(self, milestones, num_ts, num_feats, hid_dim_list, out_dim,
                 celltype='LSTM', dropout=0.0):
        """
        Simple RNN based early-classifier. RNN configuration is specified as a
        list of hidden dimensions.
        """
        super(EarlyRNNClassifier, self).__init__()
        self.celltype = celltype
        self.num_ts, self.num_feats, self.out_dim = num_ts, num_feats, out_dim
        self.hid_dim_list = hid_dim_list
        cellfn = self.__get_cell(celltype)
        inpd, cell_l = num_feats, []
        self.dropout = nn.Sequential()
        assert len(hid_dim_list) == 1, "We only support depth-1"
        if dropout > 0.0:
            raise ValueError("Dropout does not make sense for depth-1")

        for i, hidd in enumerate(hid_dim_list):
            cell = cellfn(input_size=inpd, hidden_size=hidd, num_layers=1,
                          batch_first=True)
            inpd = hidd
            cell_l.append(cell)

        self.cell_list = nn.ModuleList(cell_l)
        self.milestones = milestones
        for x in milestones:
            assert 0.0 <= x <= 1.0
        mst_idx = [int(x * num_ts) for x in milestones]
        self.milestone_idx = [min([x, num_ts-1]) for x in mst_idx]
        self.linear_list = []
        for elem in self.milestone_idx:
            lin = nn.Linear(hid_dim_list[-1], out_dim)
            self.linear_list.append(lin)
        self.linear_list = nn.ModuleList(self.linear_list)

    def __get_cell(self, celltype):
        assert celltype in ['LSTM', 'GRU']
        if celltype == 'LSTM':
            return nn.LSTM
        elif celltype == 'GRU':
            return nn.GRU

    def get_cell_out(self, output):
        '''
        Depending on the RNN celltype used, we need to do a bit of work to get
        the output states from the output of RNN layer. For example, for LSTM
        cell, the outputs are of the form
            (all layer output, (last layer h, last layer c))
        '''
        celltype = self.celltype
        if celltype == 'LSTM':
            # (all layer output, (last layer h, last layer c))
            # Further: last layer h is of shape:
            #   [num_layers, Batch size, hidden dim]
            # all_states: [batch_size, sequence_length, hid-out]
            all_states = output[0]
        elif celltype == 'GRU':
            # (all layer output, last_layer_final_stte)
            all_states = output[0]
        # Get [batch-size, sequence_length, hid-out]
        ret = all_states[:, self.milestone_idx, :]
        return ret

    def forward(self, x):
        # Initial states default to zero.
        inp = x
        cell = self.cell_list[0]
        inp, ht_ = cell(inp)
        # out: [batch-size, len(milestone_idx), hid-out]
        out_hd = self.get_cell_out((inp, ht_))
        # Unravel and multiply with linear
        ret_l = []
        for i in range(len(self.linear_list)):
            lin = self.linear_list[i]
            # all batch, ith layer, all hd
            _x = out_hd[:, i, :]
            # Add a dimension along steps for concat later
            ret_l.append(lin(_x))
        return ret_l

    def freeze(self):
        for n, p in self.named_parameters():
            p.requires_grad = False

class Composer(nn.Module):
    """
    A container/wrapper class to model arbitrary module connections.
    Modules should comprise of 
    """
    def __init__(self, non_trainables, trainable, forward_fn=None):
        super(Composer, self).__init__()
        for n, p in nn.ModuleList(non_trainables).named_parameters():
            assert p.requires_grad == False
        self.forward_fn = forward_fn
        self.ensemble = Ensemble(non_trainables)
        self.trainable = trainable

    def forward(self, x):
        if self.forward_fn is None:
            raise NotImplementedError
        ret = self.forward_fn(self, x)
        return ret


class Ensemble(nn.Module):
    """"
    Generic parallel evalution of ensemble.
    """
    def requires_grad_test(self):
        ret = []
        for mdl in self.model_list:
            for n, p in mdl.named_parameters():
                ret.append(p.requires_grad)
        return any(ret)

    def __init__(self, model_list, mixing_weights=None):
        super(Ensemble, self).__init__()
        self.model_list = nn.ModuleList(model_list)
        self.requires_grad = self.requires_grad_test()
        if mixing_weights is None:
            mixing_weights = 1.0
            if len(model_list) > 0:
                mixing_weights = [1.0/len(model_list)] * len(model_list)
        self.mixing_weights = mixing_weights


    @torch.no_grad()
    def forward(self, x):
        # Construct weight vector for each model
        assert self.model_list is not None
        res = torch.tensor(0.0)
        if len(self.model_list) == 0:
            # lg.warn("Empty ensemble found")
            return res 
        for id, model in enumerate(self.model_list):
            a = model(x)
            res = res + self.mixing_weights[id] * a
        return res
    
    def eval(self):
        for m in self.model_list:
            m.eval()

    def __len__(self):
        return len(self.model_list)

# class LogitScaler(nn.Module):
#     def __init__(self, logit_scale, multiplier=2.0):
#         """scales to logit_scale"""
#         super().__init__()
#         self.logit_scale = logit_scale
#         self.mult = multiplier
#
#     def forward(self, x):
#         return x / (self.mult * self.logit_scale)
#
# class SquashScaler(nn.Module):
#     def __init__(self, logit_scale):
#         """scales to logit_scale and then squashes using tanh"""
#         super().__init__()
#         self.logit_scale = logit_scale
#
#     def forward(self, x):
#         # approximately in [-1, 1]
#         scaled_x = x / (2 * self.logit_scale)
#         ret = torch.tanh(scaled_x)
#         return ret
