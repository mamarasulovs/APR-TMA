import numpy as np
import pytest

torch = pytest.importorskip("torch")

from apr_tma.losses import sare_penalty, total_loss
from apr_tma.metrics import evaluate_cross_domain, hter
from apr_tma.models import build_model


def test_sare_zero_for_equal_risks_and_positive_otherwise():
    labels = torch.tensor([0, 0, 1, 1])
    aug = torch.tensor([False, True, False, True])
    assert sare_penalty(torch.ones(4), labels, aug).item() == pytest.approx(0.0)
    assert sare_penalty(torch.tensor([0.1, 0.1, 2.0, 2.0]), labels, aug).item() > 0.5


def test_sare_single_group_is_zero():
    z = sare_penalty(torch.rand(3), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.bool))
    assert z.item() == 0.0


def test_total_loss_uses_gamma():
    logits = torch.tensor([2.0, -2.0, 0.5, 0.1])
    labels = torch.tensor([0, 1, 0, 1])
    aug = torch.tensor([False, True, True, False])
    full, cls, sare = total_loss(logits, labels, aug, gamma=0.1)
    assert full.item() == pytest.approx(cls.item() + 0.1 * sare.item(), rel=1e-5)
    no_sare, cls2, _ = total_loss(logits, labels, aug, use_sare=False)
    assert no_sare.item() == pytest.approx(cls2.item())


def test_hter_and_source_fixed_threshold():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert hter(scores, labels, 0.5) == 0.0
    r = evaluate_cross_domain(scores, labels, np.array([0.4, 0.6]), np.array([0, 1]))
    assert 0.0 <= r["hter"] <= 100.0 and r["auc"] == 100.0


def test_resnet18_forward_shape():
    m = build_model("resnet18", pretrained=False)
    assert m(torch.randn(2, 3, 224, 224)).shape == (2,)


def test_vit_convpass_only_adapters_and_head_train():
    pytest.importorskip("timm")
    m = build_model("vit_convpass", pretrained=False)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all(("adapt" in n) or n.startswith("head") for n in names)
    assert m(torch.randn(1, 3, 224, 224)).shape == (1,)
