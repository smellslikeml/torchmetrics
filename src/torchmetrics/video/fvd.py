# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Below is a derivative work based on the original works:
#
#   https://github.com/piergiaj/pytorch-i3d (Apache License, Version 2.0)
#     contributing the InceptionI3d architecture (MaxPool3dSamePadding, Unit3D, InceptionModule and
#     InceptionI3d), originally introduced in "Quo Vadis, Action Recognition? A New Model and the Kinetics
#     Dataset" (Carreira & Zisserman, https://arxiv.org/abs/1705.07750)
#
#   https://github.com/songweige/content-debiased-fvd (MIT License, Copyright (c) 2024 Songwei Ge)
#     contributing the video preprocessing and the feature accumulation pipeline from `cdfvd/fvd.py`
#     and `cdfvd/third_party/i3d/utils.py`
#
#   https://github.com/songweige/TATS (MIT License) hosting the Kinetics-400 pretrained I3D checkpoint
#
# adapted to torchmetrics conventions. The Frechet distance itself is not duplicated here but shared with
# `FrechetInceptionDistance` through `_compute_fid`.
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn.functional import interpolate, pad, relu

from torchmetrics.image.fid import _compute_fid
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

_I3D_WEIGHTS_URL = "https://github.com/songweige/TATS/raw/main/tats/fvd/i3d_pretrained_400.pt"
_I3D_WEIGHTS_FILENAME = "i3d_pretrained_400.pt"
_I3D_NUM_CLASSES = 400
_I3D_INPUT_SIZE = (224, 224)


def _download_i3d_weights() -> None:
    """Fetch the I3D checkpoint, used to determine upfront whether weight-dependent doctests can run."""
    NoTrainI3d()


if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["FrechetVideoDistance.plot"]


def _to_triple(value: Union[int, tuple[int, ...]]) -> tuple[int, int, int]:
    """Normalize an int or already tuple-like kernel/stride argument to a 3-tuple."""
    if isinstance(value, int):
        return (value, value, value)
    return (value[0], value[1], value[2])


class MaxPool3dSamePadding(nn.MaxPool3d):
    """3D max-pool layer that dynamically pads the input so the output size is ``ceil(input / stride)``."""

    def compute_pad(self, dim: int, size: int) -> int:
        """Compute the amount of padding needed along ``dim`` for the given input ``size``."""
        kernel_size = _to_triple(self.kernel_size)
        stride = _to_triple(self.stride)
        if size % stride[dim] == 0:
            return max(kernel_size[dim] - stride[dim], 0)
        return max(kernel_size[dim] - (size % stride[dim]), 0)

    def forward(self, x: Tensor) -> Tensor:
        """Dynamically pad the input and apply max pooling."""
        _, _, t, h, w = x.size()
        pad_t, pad_h, pad_w = self.compute_pad(0, t), self.compute_pad(1, h), self.compute_pad(2, w)

        padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_t // 2,
            pad_t - pad_t // 2,
        )
        return super().forward(pad(x, padding))


class Unit3D(nn.Module):
    """Basic 3D convolution block with optional batch normalization and activation.

    The convolution itself uses zero padding, and 'same' padding is computed dynamically in the forward pass
    so that the output size is ``ceil(input / stride)``.
    """

    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_shape: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        activation_fn: Optional[Callable[[Tensor], Tensor]] = relu,
        use_batch_norm: bool = True,
        use_bias: bool = False,
    ) -> None:
        """Initialize the convolution, normalization and activation layers.

        Args:
            in_channels: number of input channels
            output_channels: number of output channels
            kernel_shape: shape of the 3d convolution kernel
            stride: stride of the 3d convolution
            activation_fn: activation function applied after the convolution, or ``None`` for no activation
            use_batch_norm: if ``True`` a batch norm layer is applied after the convolution
            use_bias: if ``True`` the convolution uses a bias term

        """
        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation_fn = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=kernel_shape,
            stride=stride,
            padding=0,  # padding is applied dynamically in the forward pass
            bias=use_bias,
        )
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(output_channels, eps=1e-5, momentum=0.001)

    def compute_pad(self, dim: int, size: int) -> int:
        """Compute the amount of padding needed along ``dim`` for the given input ``size``."""
        if size % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        return max(self._kernel_shape[dim] - (size % self._stride[dim]), 0)

    def forward(self, x: Tensor) -> Tensor:
        """Dynamically pad the input and apply convolution, batch norm and activation."""
        _, _, t, h, w = x.size()
        pad_t, pad_h, pad_w = self.compute_pad(0, t), self.compute_pad(1, h), self.compute_pad(2, w)

        padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_t // 2,
            pad_t - pad_t // 2,
        )
        x = pad(x, padding)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """Inception block with four branches (1x1x1, 1x1x1 -> 3x3x3, 1x1x1 -> 3x3x3 and pool -> 1x1x1)."""

    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        """Initialize the four branches of the module.

        Args:
            in_channels: number of input channels
            out_channels: output channels of the six convolutions, in the order
                ``[b0, b1a, b1b, b2a, b2b, b3b]``

        """
        super().__init__()
        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=(1, 1, 1))
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=(1, 1, 1))
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=(3, 3, 3))
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=(1, 1, 1))
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=(3, 3, 3))
        self.b3a = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=(1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Run the four branches and concatenate their outputs along the channel dimension."""
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture for video feature extraction.

    The model is introduced in:
    Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
    Joao Carreira, Andrew Zisserman, https://arxiv.org/abs/1705.07750

    See also the Inception architecture, introduced in:
    Going deeper with convolutions
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan,
    Vincent Vanhoucke, Andrew Rabinovich, https://arxiv.org/abs/1409.4842

    """

    # endpoints of the model in the order they are applied in the forward pass
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    end_points: dict[str, nn.Module]

    def __init__(self, num_classes: int = 400, in_channels: int = 3, dropout_keep_prob: float = 0.5) -> None:
        """Initialize the I3D model instance.

        Args:
            num_classes: number of outputs of the logits layer (400 matches the Kinetics dataset)
            in_channels: number of input channels
            dropout_keep_prob: probability of keeping a unit in the dropout layer before logits

        """
        super().__init__()
        self._num_classes = num_classes

        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(in_channels, 64, kernel_shape=(7, 7, 7), stride=(2, 2, 2))
        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(64, 64, kernel_shape=(1, 1, 1))
        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(64, 192, kernel_shape=(3, 3, 3))
        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32])
        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64])
        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64])
        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64])
        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64])
        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64])
        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128])
        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128])
        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128])

        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            384 + 384 + 128 + 128,
            self._num_classes,
            kernel_shape=(1, 1, 1),
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
        )

        self.build()

    def build(self) -> None:
        """Register all endpoint modules as submodules so they become part of the state dict."""
        for key in self.end_points:
            self.add_module(key, self.end_points[key])

    def forward(self, x: Tensor) -> Tensor:
        """Run the network up to and including the logits layer.

        Args:
            x: input tensor of shape ``(N, C, T, H, W)``

        Returns:
            Tensor of shape ``(N, num_classes)`` with the logits averaged over the temporal dimension.

        """
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel
        x = self.logits(self.dropout(self.avg_pool(x)))
        x = x.squeeze(3).squeeze(3)
        # logits is batch x time x classes, averaged over time as in the reference implementation
        return x.mean(dim=2)


class NoTrainI3d(InceptionI3d):
    """I3D network that never leaves evaluation mode.

    The Kinetics-400 pretrained checkpoint from https://github.com/songweige/TATS (MIT licensed) is downloaded
    on first use with ``torch.hub.load_state_dict_from_url`` and afterwards served from the local cache.
    """

    def __init__(self, ckpt_path: Optional[str] = None) -> None:
        """Initialize the network and load the pretrained weights.

        Args:
            ckpt_path: optional path to a local checkpoint. If not set, the default Kinetics-400 pretrained
                checkpoint is downloaded.

        """
        super().__init__(num_classes=_I3D_NUM_CLASSES, in_channels=3)
        if ckpt_path is None:
            state_dict = torch.hub.load_state_dict_from_url(
                _I3D_WEIGHTS_URL, map_location="cpu", file_name=_I3D_WEIGHTS_FILENAME
            )
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(state_dict)
        for param in self.parameters():
            param.requires_grad = False
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool = True) -> "NoTrainI3d":
        """Force network to always be in evaluation mode."""
        return super().train(False)


# placed after NoTrainI3d is defined so that the probe can actually construct the network
if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_i3d_weights):
    __doctest_skip__ = ["FrechetVideoDistance", "FrechetVideoDistance.plot"]


def _preprocess_i3d(videos: Tensor, target_resolution: tuple[int, int] = _I3D_INPUT_SIZE) -> Tensor:
    """Resize videos to the I3D input resolution and rescale them to the ``[-1, 1]`` range.

    Ported from the reference implementation at https://github.com/songweige/content-debiased-fvd
    (``cdfvd/third_party/i3d/utils.py``, MIT licensed).

    Args:
        videos: tensor of shape ``(N, T, C, H, W)`` with values in the ``[0, 255]`` range
        target_resolution: spatial resolution the frames are resized to

    Returns:
        Tensor of shape ``(N, C, T, H', W')`` with float values in the ``[-1, 1]`` range.

    """
    n, t, c = videos.shape[:3]
    all_frames = videos.flatten(end_dim=1).float()  # (n * t, c, h, w)
    all_frames = interpolate(all_frames, size=target_resolution, mode="bilinear", align_corners=False)
    videos = all_frames.view(n, t, c, *target_resolution).transpose(1, 2).contiguous()  # (n, c, t, h, w)
    return 2.0 * videos / 255.0 - 1.0


class FrechetVideoDistance(Metric):
    r"""Calculate Fréchet Video Distance (FVD_) which is used to assess the quality of generated videos.

    .. math::
        FVD = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from the logits of
    a pretrained I3D network (Carreira & Zisserman, https://arxiv.org/abs/1705.07750) calculated on real life
    videos, and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the multivariate normal distribution estimated from the
    I3D logits calculated on generated (fake) videos. The metric was originally proposed in `FVD ref1`_. This
    implementation is a port of the reference implementation `FVD ref2`_ with the Fréchet distance itself
    shared with :class:`~torchmetrics.image.fid.FrechetInceptionDistance`.

    As input to ``update`` the metric expects mini-batches of 3-channel RGB videos of shape ``(N, T, C, H, W)``
    where ``N`` is the batch size, ``T`` the number of frames and ``C``, ``H``, ``W`` the channels, height and
    width of each frame. If argument ``normalize`` is ``True`` videos are expected to be dtype ``float`` and
    have values in the ``[0, 1]`` range, else videos are expected to have dtype ``uint8`` and values in the
    ``[0, 255]`` range. Frames are resized to 224 x 224 which is the resolution of the original training data.
    The boolean flag ``real`` determines if the videos should update the statistics of the real distribution or
    the fake distribution.

    .. note::
        Using this metric requires downloading the Kinetics-400 pretrained I3D checkpoint (approx. 50 MB) the
        first time the metric is instantiated. The checkpoint originates from https://github.com/songweige/TATS
        and is cached by ``torch.hub`` afterwards. A local checkpoint can be provided with the ``ckpt_path``
        argument to avoid the download.

    .. note::
        The temporal dimension is downsampled by the network, so at least 9 frames per video are required for
        the logits to be well defined. 16 frames per video, as used in the original FVD protocol, is
        recommended.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``videos`` (:class:`~torch.Tensor`): tensor with videos fed to the feature extractor
    - ``real`` (:class:`~bool`): bool indicating if ``videos`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``fvd`` (:class:`~torch.Tensor`): float scalar tensor with the FVD value between the real and fake
      video distributions

    Args:
        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does
            not change, the features can be cached them to avoid recomputing them which is costly. Set this to
            ``False`` if your dataset does not change.
        normalize:
            Argument for controlling input video dtype normalization:

            - True: input videos have values ranged in [0, 1] and are cast to byte tensors internally
            - False: input videos have values ranged in [0, 255] with dtype ``uint8``
        ckpt_path: optional path to a local I3D checkpoint. If not set, the Kinetics-400 pretrained checkpoint
            from https://github.com/songweige/TATS is downloaded on first use.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``normalize`` is not a ``bool``
        ValueError:
            If ``reset_real_features`` is not a ``bool``
        RuntimeError:
            If ``compute`` is called before at least two samples have been accumulated for both the real and
            the fake distribution

    Example:
        >>> import torch
        >>> from torchmetrics.video.fvd import FrechetVideoDistance
        >>> fvd = FrechetVideoDistance()
        >>> # generate two slightly overlapping video intensity distributions
        >>> videos_dist1 = torch.randint(0, 200, (8, 16, 3, 64, 64), dtype=torch.uint8)
        >>> videos_dist2 = torch.randint(100, 255, (8, 16, 3, 64, 64), dtype=torch.uint8)
        >>> fvd.update(videos_dist1, real=True)
        >>> fvd.update(videos_dist2, real=False)
        >>> fvd_score = fvd.compute()
        >>> fvd_score >= 0
        True

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    i3d: Module
    feature_network: str = "i3d"

    def __init__(
        self,
        reset_real_features: bool = True,
        normalize: bool = False,
        ckpt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        self.i3d = NoTrainI3d(ckpt_path=ckpt_path)

        num_features = _I3D_NUM_CLASSES
        mx_num_feats = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def update(self, videos: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            videos: Input video tensors of shape ``(N, T, C, H, W)``. If ``normalize`` is ``True`` the values
                are expected to be floats in the ``[0, 1]`` range, else dtype ``uint8`` with values in the
                ``[0, 255]`` range.
            real: Whether given videos are real or fake.

        """
        videos = (videos * 255).byte() if self.normalize else videos
        features = self.i3d(_preprocess_i3d(videos))
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += videos.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += videos.shape[0]

    def compute(self) -> Tensor:
        """Calculate FVD score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FVD")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these
                results. If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.video.fvd import FrechetVideoDistance
            >>> metric = FrechetVideoDistance()
            >>> metric.update(torch.randint(0, 200, (8, 16, 3, 64, 64), dtype=torch.uint8), real=True)
            >>> metric.update(torch.randint(100, 255, (8, 16, 3, 64, 64), dtype=torch.uint8), real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.video.fvd import FrechetVideoDistance
            >>> metric = FrechetVideoDistance()
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(torch.randint(0, 200, (8, 16, 3, 64, 64), dtype=torch.uint8), real=True)
            ...     metric.update(torch.randint(100, 255, (8, 16, 3, 64, 64), dtype=torch.uint8), real=False)
            ...     values.append(metric.compute())
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
