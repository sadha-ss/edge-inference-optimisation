"""Model, profiling, pruning and quantisation tests (require PyTorch)."""
import pytest

torch = pytest.importorskip("torch")

from edgeopt.models import TASK_SPECS, ModelConfig, build_model          # noqa: E402
from edgeopt.profile import count_params, profile_model                    # noqa: E402
from edgeopt.prune import MagnitudePruner, prune_structured                # noqa: E402
from edgeopt.train import distillation_loss                                # noqa: E402

CASES = [(f, t) for f in ("dscnn", "resnet", "transformer") for t in TASK_SPECS]


@pytest.mark.parametrize("family,task", CASES)
def test_forward_shapes_and_scaling(family, task):
    res = TASK_SPECS[task]["resolutions"][0]
    sizes = []
    for w in (0.25, 1.0):
        cfg = ModelConfig(family, task, w, res)
        m = build_model(cfg).eval()
        out = m(torch.zeros(cfg.input_shape(2)))
        assert out.shape == (2, cfg.n_classes)
        sizes.append(count_params(m))
    assert sizes[0] < sizes[1]


@pytest.mark.parametrize("family,task", CASES)
def test_structured_pruning_shrinks_and_runs(family, task):
    cfg = ModelConfig(family, task, 1.0, TASK_SPECS[task]["resolutions"][0])
    m = build_model(cfg).eval()
    p = prune_structured(m, 0.5).eval()
    assert count_params(p) < count_params(m)
    assert p(torch.zeros(cfg.input_shape(2))).shape == (2, cfg.n_classes)
    ex = torch.zeros(cfg.input_shape(1))
    assert profile_model(p, ex)["macs"] < profile_model(m, ex)["macs"]


def test_profile_macs_match_formula():
    cfg = ModelConfig("transformer", "kws", 1.0, 64)
    m = build_model(cfg).eval()
    prof = profile_model(m, torch.zeros(cfg.input_shape(1)))
    d, n, layers = 128, 64, 4
    expected_blocks = layers * (12 * n * d * d + 2 * n * n * d)      # Eq. 9
    assert prof["macs"] >= expected_blocks


def test_unstructured_pruner_reaches_sparsity():
    m = build_model(ModelConfig("dscnn", "kws", 1.0, 49))
    pr = MagnitudePruner(m, 0.7, 0, 2)
    for e in range(3):
        pr.update(e)
    assert 0.65 < pr.sparsity() < 0.75


def test_distillation_loss_zero_when_teacher_matches_and_lambda_one():
    logits = torch.randn(8, 5)
    y = torch.randint(0, 5, (8,))
    assert distillation_loss(logits, logits.clone(), y, lam=1.0).item() < 1e-5
