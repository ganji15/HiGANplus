import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none', pad_type='reflect', sn=False, dropout=0.):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type=pad_type, norm=norm,
                                  activation=activation, activation_first=True, sn=sn)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type=pad_type, norm=norm,
                                  activation=activation, activation_first=True, sn=sn)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False, sn=sn)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = Identity()

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.dropout(dx)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class TimeBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.nn = block

    def forward(self, tmaps):
        bz, t = tmaps.size(0), tmaps.size(1)
        map_size = tmaps.size()[2:]
        flatten_maps = tmaps.view(bz * t, *map_size)
        flatten_maps = self.nn(flatten_maps)
        tmaps = flatten_maps.view(bz, t, *map_size)
        return tmaps


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, groups=1,
                 sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if padding == 0:
            self.pad = Identity()
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(4, norm_dim, 0.8)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'iln':
            self.norm = InstanceLayerNorm2d(norm_dim)
        elif norm == 'adailn':
            self.norm = AdaptiveInstanceLayerNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if sn:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias, groups=groups))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias, groups=groups)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3, norm='none', activ='relu'):
        super(MLP, self).__init__()

        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
        pass

    def forward(self, x):
        return x


class DeepLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being unidirectional."""
    def __init__(
        self, input_size, hidden_size, n_layers=2,
        dropout=0., batch_first=True
    ):
        """Initialize params."""
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.n_layer = n_layers

        self.lstm = nn.LSTM(input_size,
                            self.hidden_size,
                            n_layers,
                            bidirectional=False,
                            batch_first=True,
                            dropout=self.dropout
                            )

        self.lstm.flatten_parameters()

    def forward(self, x, x_len=None):
        """Propogate input forward through the network."""
        init_hidden = self.get_init_state(x.size(0), x.device)
        out, _ = self.lstm(x, init_hidden)
        return out

    def get_init_state(self, batch_size, device):
        """Get cell states and hidden states."""
        deepth = self.n_layer
        hidden_dim = self.hidden_size

        h0_encoder_bi = torch.zeros(
            deepth,
            batch_size,
            hidden_dim, requires_grad=False)
        c0_encoder_bi = torch.zeros(
            deepth,
            batch_size,
            hidden_dim, requires_grad=False)
        return h0_encoder_bi.to(device), c0_encoder_bi.to(device)



class DeepGRU(nn.Module):
    r"""A Deep LSTM with the first layer being unidirectional."""
    def __init__(
        self, input_size, hidden_size, n_layers=2,
        dropout=0., batch_first=True
    ):
        """Initialize params."""
        super(DeepGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.n_layer = n_layers

        self.lstm = nn.GRU(input_size,
                            self.hidden_size,
                            n_layers,
                            bidirectional=False,
                            batch_first=True,
                            dropout=self.dropout
                            )

        self.lstm.flatten_parameters()

    def forward(self, x, x_len=None):
        """Propogate input forward through the network."""
        init_hidden = self.get_init_state(x.size(0), x.device)
        out, _ = self.lstm(x, init_hidden)
        return out

    def get_init_state(self, batch_size, device):
        """Get cell states and hidden states."""
        deepth = self.n_layer
        hidden_dim = self.hidden_size

        h0_encoder_bi = torch.zeros(
            deepth,
            batch_size,
            hidden_dim, requires_grad=False)
        return h0_encoder_bi.to(device)



class DeepBLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being bidirectional."""
    def __init__(
        self, input_size, hidden_size, n_layers=2,
        dropout=0., batch_first=True, bidirectional=True
    ):
        """Initialize params."""
        super(DeepBLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.n_layer = n_layers
        self.bidirectional = bidirectional
        hidden_split = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(input_size,
                            self.hidden_size // hidden_split,
                            n_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=self.dropout
                            )

        self.lstm.flatten_parameters()

    def forward(self, x, x_len):
        """Propogate input forward through the network."""
        self.lstm.flatten_parameters()
        x_pack = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        init_hidden = self.get_init_state(x.size(0), x.device)
        out_pack, _ = self.lstm(x_pack, init_hidden)
        out, out_len = pad_packed_sequence(out_pack, batch_first=self.batch_first)
        return out

    def get_init_state(self, batch_size, device):
        """Get cell states and hidden states."""
        deepth = self.n_layer
        hidden_dim = self.hidden_size
        if self.bidirectional:
            deepth *= 2
            hidden_dim //= 2

        h0_encoder_bi = torch.zeros(
            deepth,
            batch_size,
            hidden_dim, requires_grad=False)
        c0_encoder_bi = torch.zeros(
            deepth,
            batch_size,
            hidden_dim, requires_grad=False)
        return h0_encoder_bi.to(device), c0_encoder_bi.to(device)


class CosMargin(nn.Module):
    def __init__(self, in_size, out_size, s=None, m=0.):
        super(CosMargin, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.W = nn.Parameter(torch.randn(out_size, in_size), requires_grad=True)
        self.W.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.s = nn.Parameter(torch.randn(1,), requires_grad=True) if s is None else s
        self.m = m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.W))
        if label is not None and math.fabs(self.m) > 1e-6:
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
            output = (cosine - one_hot * self.m) * self.s
        else:
            output = cosine * self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(in_size={}, out_size={}, s={}, m={})'.format(
                    self.in_size, self.out_size,
                    'learn' if isinstance(self.s, nn.Parameter) else self.s,
                    'learn' if isinstance(self.m, nn.Parameter) else self.m)


class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        nn.init.ones_(self.weights.weight.data)
        nn.init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalBatchNorm2d, self).forward(
                     input, weight, bias)


class StyleBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, in_features, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(StyleBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Linear(in_features, num_features)
        self.biases = nn.Linear(in_features, num_features)

        self._initialize()

    def _initialize(self):
        nn.init.ones_(self.weights.weight.data)
        nn.init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(StyleBatchNorm2d, self).forward(
                     input, weight, bias)


class ConditionalResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, dim_style, w_hpf=0,
                 actv='relu', pad_type='reflect'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.learned_sc = dim_in != dim_out

        if actv == 'relu':
            self.actv = nn.ReLU(inplace=False)
        elif actv == 'lrelu':
            self.actv = nn.LeakyReLU(0.2, inplace=False)
        elif actv == 'tanh':
            self.actv = nn.Tanh()
        elif actv == 'none':
            self.actv = None
        else:
            assert 0, "Unsupported activation: {}".format(actv)

        self.conv1 = Conv2dBlock(dim_in, dim_out, 3, 1, 1, pad_type=pad_type,
                                 activation='none')

        self.conv2 = Conv2dBlock(dim_out, dim_out, 3, 1, 1, pad_type=pad_type,
                                 activation='none')
        self.norm1 = StyleBatchNorm2d(dim_style, dim_out)
        self.norm2 = StyleBatchNorm2d(dim_style, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class InstanceLayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(InstanceLayerNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3)
            self.rho[:, :, 2].data.fill_(3)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(1)
            self.rho[:, :, 1].data.fill_(3.2)

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), \
                                  torch.var(input, dim=[0, 2, 3], keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        # print('ILN-weight:{} out:{}'.format(self.gamma.size(), out.size()))
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        return out


class AdaptiveInstanceLayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(AdaptiveInstanceLayerNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features

        if self.using_bn:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:, :, 0].data.fill_(3)
            self.rho[:, :, 1].data.fill_(1)
            self.rho[:, :, 2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1, 1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = nn.Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:, :, 0].data.fill_(3.2)
            self.rho[:, :, 1].data.fill_(1)

        self.weight = None
        self.bias = None

    def forward(self, input):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaILN weight first"

        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)

        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3],
                                                                                            keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_2 = rho[:, :, 2]

            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_2 = rho_2.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:, :, 0]
            rho_1 = rho[:, :, 1]
            rho_0 = rho_0.view(1, self.num_features, 1, 1)
            rho_1 = rho_1.view(1, self.num_features, 1, 1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln
        # print('AdaILN-weight:{} out:{}'.format(self.weight.size(), out.size()))
        out = out * self.weight.unsqueeze(2).unsqueeze(3) + self.bias.unsqueeze(2).unsqueeze(3)
        return out


def assign_adaptive_norm_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ in ["AdaptiveInstanceNorm2d",
                                    "AdaptiveInstanceLayerNorm2d"]:
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
            else:
                m.bias = mean.contiguous()
                m.weight = std.contiguous()

            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adaptive_norm_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adaptive_norm_params = 0
    for m in model.modules():
        if m.__class__.__name__ in ["AdaptiveInstanceNorm2d",
                                    "AdaptiveInstanceLayerNorm2d"]:
            num_adaptive_norm_params += 2 * m.num_features
    return num_adaptive_norm_params

