import json

from pv_anomaly.train import train


def _run(cfg, tmp_path, name, seed):
    cfg.name, cfg.out_dir, cfg.train_seed = name, tmp_path / "runs", seed
    cfg.epochs, cfg.device, cfg.batch_size = 2, "cpu", 2
    cfg.encoder, cfg.encoder_weights, cfg.num_workers = "resnet18", None, 2
    metrics = train(cfg)
    history = json.loads((cfg.run_dir / "history.json").read_text())
    return metrics["iou"], [h["train"] for h in history]


def test_same_training_seed_repeats_the_run(cfg, tmp_path):
    a = _run(cfg, tmp_path, "a", 7)
    b = _run(cfg, tmp_path, "b", 7)
    assert a == b


def test_different_training_seeds_differ(cfg, tmp_path):
    a = _run(cfg, tmp_path, "a", 7)
    b = _run(cfg, tmp_path, "b", 8)
    assert a[1] != b[1]
