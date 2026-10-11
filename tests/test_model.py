import numpy as np
import pytest
import torch

from pv_anomaly.config import Config
from pv_anomaly.losses import CEDiceLoss, median_frequency_weights
from pv_anomaly.metrics import evaluate, json_safe, report, size_recall

# --- config ---------------------------------------------------------------


def test_config_round_trips_through_yaml(tmp_path):
    cfg = Config(name="x", tile=256, encoder="resnet18")
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert Config.load(path).as_dict() == cfg.as_dict()


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\ntyop: 3\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(path)


def test_run_dir_reflects_split_mode():
    assert Config(split_mode="predefined").run_dir.name.endswith("_predefined")
    assert Config(split_mode="cv", fold=3).run_dir.name.endswith("_fold3")


# --- loss -----------------------------------------------------------------


def test_weights_upweight_the_rare_class():
    assert median_frequency_weights(np.array([99_000, 1_000]))[1] > 1.0


def test_absent_class_gets_zero_weight_not_inf():
    weights = median_frequency_weights(np.array([100, 0, 50]))
    assert weights[1] == 0.0 and torch.isfinite(weights).all()


def test_weights_respect_the_clip():
    assert median_frequency_weights(np.array([10**9, 1]), clip=5.0).max() <= 5.0


def test_weights_reject_empty_counts():
    with pytest.raises(ValueError):
        median_frequency_weights(np.array([0, 0]))


def test_loss_is_near_zero_for_a_perfect_prediction():
    target = torch.zeros(2, 16, 16, dtype=torch.long)
    target[:, 4:12, 4:12] = 1
    logits = torch.full((2, 2, 16, 16), -20.0)
    logits[:, 0][target == 0] = 20.0
    logits[:, 1][target == 1] = 20.0
    assert CEDiceLoss(2)(logits, target).item() < 1e-3


def test_loss_skips_classes_absent_from_the_batch():
    logits = torch.randn(1, 3, 8, 8)
    assert torch.isfinite(CEDiceLoss(3)(logits, torch.zeros(1, 8, 8, dtype=torch.long)))


def test_loss_backprops():
    logits = torch.randn(2, 2, 16, 16, requires_grad=True)
    CEDiceLoss(2, torch.tensor([1.0, 5.0]))(
        logits, torch.randint(0, 2, (2, 16, 16))
    ).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("kwargs", [{"alpha": 1.5}, {"weights": torch.ones(3)}])
def test_loss_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        CEDiceLoss(2, **kwargs)


# --- metrics --------------------------------------------------------------


def test_perfect_prediction_scores_one():
    target = np.array([[0, 1], [1, 0]], np.uint8)
    m = evaluate([(target, target, target.astype(np.float32))], 2)
    assert m["mean_iou"] == pytest.approx(1.0) and m["auprc"] == pytest.approx(1.0)


def test_iou_is_nan_for_a_class_that_never_appears():
    zeros = np.zeros((4, 4), np.uint8)
    assert np.isnan(evaluate([(zeros, zeros, zeros.astype(np.float32))], 3)["iou"][2])


def test_auprc_is_none_on_a_single_class_fold():
    """Undefined with no positives -- must be None, not a crash."""
    zeros = np.zeros((16, 16), np.uint8)
    assert evaluate([(zeros, zeros, zeros.astype(np.float32))], 2)["auprc"] is None


def test_size_recall_buckets_regions_by_area():
    target = np.zeros((200, 200), np.uint8)
    target[10:13, 10:13] = 1  # 9 px
    target[50:110, 50:110] = 1  # 3600 px
    buckets = [b for b in size_recall([target], [target]) if b["n"]]
    assert sum(b["n"] for b in buckets) == 2
    assert all(b["recall"] == 1.0 for b in buckets)


def test_size_recall_records_misses():
    target = np.zeros((100, 100), np.uint8)
    target[10:14, 10:14] = 1
    hit = next(b for b in size_recall([target], [np.zeros_like(target)]) if b["n"])
    assert hit["recall"] == 0.0


def test_report_renders_every_section():
    target = np.array([[0, 1], [1, 0]], np.uint8)
    text = report(
        evaluate([(target, target, target.astype(np.float32))], 2),
        ["background", "defect1"],
    )
    assert "mean IoU" in text and "macro F1" in text and "pixel AUPRC" in text


def test_evaluate_consumes_a_generator_without_retaining_images():
    """The test set is ~2 GB of arrays; evaluate must stream, not collect."""
    seen = []

    def stream():
        for _ in range(5):
            target = np.zeros((64, 64), np.uint8)
            target[10:20, 10:20] = 1
            seen.append(target)
            yield target, target, target.astype(np.float32)

    m = evaluate(stream(), 2)
    assert len(seen) == 5  # generator was fully consumed
    assert m["mean_iou"] == pytest.approx(1.0)
    assert sum(b["n"] for b in m["size_recall"]) == 5  # one region per image


def test_score_sampling_matches_the_exact_value():
    """Subsampled AUPRC/AUROC must agree with scoring every pixel."""
    rng = np.random.default_rng(0)
    images = []
    for _ in range(8):
        target = (rng.random((256, 256)) > 0.97).astype(np.uint8)
        score = np.clip(target + rng.normal(0, 0.3, target.shape), 0, 1).astype(
            np.float32
        )
        images.append((target, target, score))

    exact = evaluate(iter(images), 2, score_pixels=0xFFFFFF)
    sampled = evaluate(iter(images), 2, score_pixels=20_000)
    assert sampled["auprc"] == pytest.approx(exact["auprc"], abs=0.02)
    assert sampled["auroc"] == pytest.approx(exact["auroc"], abs=0.02)


def test_json_safe_removes_nan_and_inf():
    """Python writes nan and inf into JSON; browsers and most readers reject it."""
    import json

    zeros = np.zeros((16, 16), np.uint8)
    metrics = evaluate([(zeros, zeros, zeros.astype(np.float32))], 3)
    assert "NaN" in json.dumps(metrics)  # what we are protecting against

    text = json.dumps(json_safe(metrics))
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["iou"][2] is None
