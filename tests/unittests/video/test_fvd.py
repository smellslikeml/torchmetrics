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
import pickle
from collections.abc import Callable

import pytest
import torch
from torch.nn import Module

from torchmetrics.image.fid import _compute_fid
from torchmetrics.video.fvd import FrechetVideoDistance, NoTrainI3d, _preprocess_i3d
from unittests._helpers import seed_all

seed_all(42)

# small but valid video format: 4 videos with the minimum of 9 frames each (16 frames is the FVD standard)
BATCH_SIZE, NUM_FRAMES = 4, 9


@pytest.fixture(scope="module")
def fvd_factory() -> Callable[[], FrechetVideoDistance]:
    """Return a factory for the metric, skipping weight-dependent tests when the checkpoint is unavailable.

    Constructing the metric requires the Kinetics-400 pretrained I3D checkpoint, which is downloaded from
    https://github.com/songweige/TATS on first use. The probe construction runs once per module and any
    following constructions are served from the torch.hub cache.

    """
    try:
        FrechetVideoDistance()
    except Exception as error:
        pytest.skip(f"FVD tests require the I3D checkpoint, which could not be fetched: {error}")
    return FrechetVideoDistance


def test_preprocess_i3d():
    """Preprocessing resizes to the I3D input resolution and rescales to the [-1, 1] range."""
    videos = torch.randint(0, 255, (2, 16, 3, 32, 24), dtype=torch.uint8)
    processed = _preprocess_i3d(videos)

    assert processed.shape == (2, 3, 16, 224, 224)
    assert processed.dtype == torch.float32
    assert processed.min() >= -1.0
    assert processed.max() <= 1.0


def test_fvd_raises_errors():
    """Argument validation errors are raised before the checkpoint is fetched."""
    with pytest.raises(ValueError, match="Argument `normalize` expected to be a bool"):
        _ = FrechetVideoDistance(normalize=1)

    with pytest.raises(ValueError, match="Argument `reset_real_features` expected to be a bool"):
        _ = FrechetVideoDistance(reset_real_features=1)


def test_no_train(fvd_factory):
    """Assert that metric never leaves evaluation mode."""
    class MyModel(Module):
        def __init__(self) -> None:
            super().__init__()
            self.metric = fvd_factory()

        def forward(self, x):
            return x

    model = MyModel()
    model.train()
    assert model.training
    assert not model.metric.i3d.training, "FVD metric was changed to training mode which should not happen"


def test_fvd_pickle(fvd_factory):
    """Assert that we can initialize the metric and pickle it."""
    metric = fvd_factory()
    assert metric

    # verify metrics work after being loaded from pickled state
    pickled_metric = pickle.dumps(metric)
    metric = pickle.loads(pickled_metric)
    assert metric


def test_fvd_smoke(fvd_factory):
    """Smoke test that the computed FVD score is a finite, non-negative scalar."""
    metric = fvd_factory()

    videos_dist1 = torch.randint(0, 200, (BATCH_SIZE, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8)
    videos_dist2 = torch.randint(100, 255, (BATCH_SIZE, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8)
    metric.update(videos_dist1, real=True)
    metric.update(videos_dist2, real=False)

    val = metric.compute()
    assert val.ndim == 0, "FVD should be a scalar"
    assert torch.isfinite(val), "FVD should be finite"
    assert val >= 0, "FVD should be non-negative"


def test_fvd_same_input(fvd_factory):
    """If real and fake are updated on the same data the fvd score should be 0."""
    metric = fvd_factory()

    seed_all(42)
    for _ in range(2):
        videos = torch.randint(0, 255, (BATCH_SIZE, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8)
        metric.update(videos, real=True)
        metric.update(videos, real=False)

    assert torch.allclose(metric.real_features_sum, metric.fake_features_sum)
    assert torch.allclose(metric.real_features_cov_sum, metric.fake_features_cov_sum)
    assert torch.allclose(metric.real_features_num_samples, metric.fake_features_num_samples)

    val = metric.compute()
    assert torch.allclose(val, torch.zeros_like(val), atol=1e-3)


def test_fvd_matches_compute_fid(fvd_factory):
    """Accumulated statistics must give the same score as FID's `_compute_fid` on the raw I3D features.

    This checks the feature accumulation and the reuse of the Fréchet distance from
    `torchmetrics.image.fid` against a direct computation from the extracted features. Full numerical
    parity of the port with the reference `cdfvd` implementation (https://github.com/songweige/
    content-debiased-fvd) was additionally verified outside the test suite: on a fixed set of uint8
    videos of shape ``(N, T, H, W, C)``, the vendored I3D produces bit-identical features to the
    reference ``InceptionI3d`` + ``preprocess_i3d``, and the resulting FVD agrees with the reference
    ``frechet_distance`` within the float tolerance of the two different matrix-square-root
    implementations (abs. diff 0.005 on a score of 519.8, i.e. rel. 1e-5).

    """
    metric = fvd_factory()
    seed_all(42)

    real = torch.randint(0, 200, (BATCH_SIZE, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8)
    fake = torch.randint(100, 255, (BATCH_SIZE, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8)
    metric.update(real, real=True)
    metric.update(fake, real=False)

    with torch.no_grad():
        real_features = metric.i3d(_preprocess_i3d(real)).double()
        fake_features = metric.i3d(_preprocess_i3d(fake)).double()

    mean_real, mean_fake = real_features.mean(dim=0), fake_features.mean(dim=0)
    cov_real, cov_fake = torch.cov(real_features.t()), torch.cov(fake_features.t())
    expected = _compute_fid(mean_real, cov_real, mean_fake, cov_fake)

    assert torch.allclose(metric.compute().double(), expected, atol=1e-6)


def test_not_enough_samples(fvd_factory):
    """Test that an error is raised if not enough samples were provided."""
    videos = torch.randint(0, 255, (1, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8)
    metric = fvd_factory()
    metric.update(videos, real=True)
    metric.update(videos, real=False)
    with pytest.raises(
        RuntimeError, match="More than one sample is required for both the real and fake distributed to compute FVD"
    ):
        metric.compute()


@pytest.mark.parametrize("reset_real_features", [True, False])
def test_reset_real_features_arg(fvd_factory, reset_real_features):
    """Test that `reset_real_features` argument works as expected."""
    metric = fvd_factory(reset_real_features=reset_real_features)

    metric.update(torch.randint(0, 180, (2, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8), real=True)
    metric.update(torch.randint(0, 180, (2, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8), real=False)

    assert metric.real_features_num_samples == 2
    assert metric.real_features_sum.shape == torch.Size([400])
    assert metric.real_features_cov_sum.shape == torch.Size([400, 400])

    assert metric.fake_features_num_samples == 2
    assert metric.fake_features_sum.shape == torch.Size([400])
    assert metric.fake_features_cov_sum.shape == torch.Size([400, 400])

    metric.reset()

    # fake features should always reset
    assert metric.fake_features_num_samples == 0

    if reset_real_features:
        assert metric.real_features_num_samples == 0
    else:
        assert metric.real_features_num_samples == 2
        assert metric.real_features_sum.shape == torch.Size([400])
        assert metric.real_features_cov_sum.shape == torch.Size([400, 400])


@pytest.mark.parametrize("normalize", [True, False])
def test_normalize_arg(fvd_factory, normalize):
    """Test that normalize argument works as expected."""
    if normalize:
        videos = torch.rand(2, NUM_FRAMES, 3, 64, 64)
        metric = fvd_factory(normalize=True)
        metric.update(videos, real=True)
        assert metric.real_features_num_samples == 2
    else:
        videos = torch.randint(0, 255, (2, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8)
        metric = fvd_factory(normalize=False)
        metric.update(videos, real=True)
        assert metric.real_features_num_samples == 2


def test_no_train_i3d_local_checkpoint(fvd_factory, tmp_path):
    """A locally saved checkpoint can be loaded through ``ckpt_path`` instead of downloading."""
    checkpoint = tmp_path / "i3d.pt"
    torch.save(NoTrainI3d().state_dict(), checkpoint)

    metric = FrechetVideoDistance(ckpt_path=str(checkpoint))
    assert metric.real_features_num_samples == 0

    metric.update(torch.randint(0, 255, (2, NUM_FRAMES, 3, 64, 64), dtype=torch.uint8), real=True)
    assert metric.real_features_num_samples == 2
