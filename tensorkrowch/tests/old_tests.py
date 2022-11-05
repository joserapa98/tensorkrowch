"""
import tensorkrowch as tn

node1 = tn.ParamNode(shape=(2, 5, 3),
                     axes_names=('left', 'input', 'right'),
                     name='node1',
                     network=None,
                     param_edges=False,
                     tensor=None,
                     init_method='ones')

node2 = tn.ParamNode(shape=(3, 5, 4),
                     axes_names=('left', 'input', 'right'),
                     name='node2',
                     network=None,
                     param_edges=False,
                     tensor=None,
                     init_method='ones')

print(node1.edges)
print(node2.edges)
print()

node1['right'] ^ node2['left']
print(node1.edges)
print(node2.edges)

node3 = tn.batched_contract_between(node1, node2, node1['input'], node2['input'])
print(node3)
print(node3.axes_names)
print(node3.edges)
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

import timeit


class Conv2d(nn.Module):
    def __init__(
            self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.n_channels = n_channels
        self.weights = nn.Parameter(
            torch.Tensor(self.out_channels, self.n_channels, self.kernel_size ** 2)
        )

    def __repr__(self):
        return (
            f"Conv2d(n_channels={self.n_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size})"
        )

    def forward(self, x):
        width = self.calculate_new_width(x)
        height = self.calculate_new_height(x)
        windows = self.calculate_windows(x)

        result = torch.zeros(
            [x.shape[0] * self.out_channels, width, height],
            dtype=torch.float32, device=x.device
        )

        for channel in range(x.shape[1]):
            for i_conv_n in range(self.out_channels):
                xx = torch.matmul(windows[channel], self.weights[i_conv_n][channel])
                xx = xx.view((-1, width, height))

                xx_stride = slice(i_conv_n * xx.shape[0], (i_conv_n + 1) * xx.shape[0])
                result[xx_stride] += xx

        result = result.view((x.shape[0], self.out_channels, width, height))
        return result

    def calculate_windows(self, x):
        windows = F.unfold(
            x,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            stride=(self.stride, self.stride)
        )

        windows = (windows
                   .transpose(1, 2)
                   .contiguous().view((-1, x.shape[1], int(self.kernel_size ** 2)))
                   .transpose(0, 1)
                   )
        return windows

    def calculate_new_width(self, x):
        return (
                       (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1

    def calculate_new_height(self, x):
        return (
                       (x.shape[3] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1


class Conv2d_jit(jit.ScriptModule):
    def __init__(
            self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.n_channels = n_channels
        self.weights = nn.Parameter(
            torch.Tensor(self.out_channels, self.n_channels, self.kernel_size ** 2)
        )

    def __repr__(self):
        return (
            f"Conv2d(n_channels={self.n_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size})"
        )

    @jit.script_method
    def forward(self, x):
        width = self.calculate_new_width(x)
        height = self.calculate_new_height(x)
        windows = self.calculate_windows(x)

        result = torch.zeros(
            [x.shape[0] * self.out_channels, width, height],
            dtype=torch.float32, device=x.device
        )

        for channel in range(x.shape[1]):
            for i_conv_n in range(self.out_channels):
                xx = torch.matmul(windows[channel], self.weights[i_conv_n][channel])
                xx = xx.view((-1, width, height))

                xx_stride = slice(i_conv_n * xx.shape[0], (i_conv_n + 1) * xx.shape[0])
                result[xx_stride] += xx

        result = result.view((x.shape[0], self.out_channels, width, height))
        return result

    def calculate_windows(self, x):
        windows = F.unfold(
            x,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=(self.padding, self.padding),
            dilation=(self.dilation, self.dilation),
            stride=(self.stride, self.stride)
        )

        windows = (windows
                   .transpose(1, 2)
                   .contiguous().view((-1, x.shape[1], int(self.kernel_size ** 2)))
                   .transpose(0, 1)
                   )
        return windows

    def calculate_new_width(self, x):
        return (
                       (x.shape[2] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1

    def calculate_new_height(self, x):
        return (
                       (x.shape[3] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                       // self.stride
               ) + 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Conv2d:
# -------
z = torch.randint(0, 255, (1, 3, 512, 512), device=device) / 255
conv = Conv2d(3, 16, 3)
conv.to(device)

start = timeit.default_timer()

out = conv(z)
out.mean().backward()

stop = timeit.default_timer()

print('Time: ', stop - start)


# Conv2d_jit:
# -----------
z = torch.randint(0, 255, (1, 3, 512, 512), device=device) / 255
conv = Conv2d_jit(3, 16, 3)
conv.to(device)

start = timeit.default_timer()

out = conv(z)
out.mean().backward()

stop = timeit.default_timer()

print('Time: ', stop - start)


# nn.Conv2d:
# -----------
z = torch.randint(0, 255, (1, 3, 512, 512), device=device) / 255
conv = nn.Conv2d(3, 16, 3)
conv.to(device)

start = timeit.default_timer()

out = conv(z)
out.mean().backward()

stop = timeit.default_timer()

print('Time: ', stop - start)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 3, 3)
        self.conv2 = Conv2d(3, 9, 3)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class NN_jit(jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 3, 3)
        self.conv2 = Conv2d(3, 9, 3)

    @jit.script_method
    def forward(self, x):
        return self.conv2(self.conv1(x))


class NN_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 9, 3)

    def forward(self, x):
        return self.conv2(self.conv1(x))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Conv2d:
# -------
z = torch.randint(0, 255, (1, 1, 512, 512), device=device) / 255
conv = NN()
conv.to(device)

start = timeit.default_timer()

out = conv(z)
out.mean().backward()

stop = timeit.default_timer()

print('Time: ', stop - start)


# Conv2d_jit:
# -----------
z = torch.randint(0, 255, (1, 1, 512, 512), device=device) / 255
conv = NN_jit()
conv.to(device)

start = timeit.default_timer()

out = conv(z)
out.mean().backward()

stop = timeit.default_timer()

print('Time: ', stop - start)


# nn.Conv2d:
# -----------
z = torch.randint(0, 255, (1, 1, 512, 512), device=device) / 255
conv = NN_nn()
conv.to(device)

start = timeit.default_timer()

out = conv(z)
out.mean().backward()

stop = timeit.default_timer()

print('Time: ', stop - start)



# !!!!!!!
# Very efficient mplementation of Conv2d (not as much as nn.Conv2d)
batch_size = 16
channels = 5
h, w = 32, 32
image = torch.randn(batch_size, channels, h, w) # input image

kh, kw = 3, 3 # kernel size
dh, dw = 3, 3 # stride

# Create conv
start1 = timeit.default_timer()
conv = nn.Conv2d(5, 7, (kh, kw), stride=(dh, dw), bias=False)
filt = conv.weight
stop1 = timeit.default_timer()

# Manual approach
@jit.script
def myconv():
    patches = image.unfold(2, kh, dh).unfold(3, kw, dw)
    print(patches.shape) # batch_size, channels, h_windows, w_windows, kh, kw

    patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)
    print(patches.shape) # batch_size, channels, windows, kh, kw

    nb_windows = patches.size(2)

    # Now we have to shift the windows into the batch dimension.
    # Maybe there is another way without .permute, but this should work
    patches = patches.permute(0, 2, 1, 3, 4)
    print(patches.shape) # batch_size, nb_windows, channels, kh, kw

    # Calculate the conv operation manually
    res = (patches.unsqueeze(2) * filt.unsqueeze(0).unsqueeze(1)).sum([3, 4, 5])
    print(res.shape) # batch_size, output_pixels, out_channels
    res = res.permute(0, 2, 1) # batch_size, out_channels, output_pixels
    # assuming h = w
    h = w = int(res.size(2)**0.5)
    res = res.view(batch_size, -1, h, w)
    return res

start2 = timeit.default_timer()
res = myconv()
stop2 = timeit.default_timer()
print('Time: ', stop2 - start2)


start3 = timeit.default_timer()
# Module approach
out = conv(image)
stop3 = timeit.default_timer()
print('Time: ', stop1 - start1 + stop3 - start3)
"""

import torch
import timeit

a = torch.rand(1, 1, 16, 2, 16, 2, 16, 2, 2, 2, 2)
b = torch.rand(729, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2)

start = timeit.default_timer()

stop = timeit.default_timer()
print('Time: ', stop - start)
