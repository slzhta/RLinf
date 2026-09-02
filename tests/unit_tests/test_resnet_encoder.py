import torch

from rlinf.models.embodiment.modules.resnet_utils import ResNetEncoder


def test_zero_dropout_keeps_encoder_output_mode_invariant(monkeypatch):
    monkeypatch.setattr(
        ResNetEncoder,
        "_load_pretrained_weights",
        lambda self: None,
    )
    sample = torch.randn(2, 3, 128, 128)
    encoder = ResNetEncoder(
        sample,
        encoder_cfg={"ckpt_path": "unused", "dropout": 0.0},
    )

    encoder.eval()
    eval_output = encoder(sample)
    encoder.train()
    train_output = encoder(sample)

    torch.testing.assert_close(train_output, eval_output)


def test_trainable_backbone_receives_gradients(monkeypatch):
    monkeypatch.setattr(
        ResNetEncoder,
        "_load_pretrained_weights",
        lambda self: None,
    )
    sample = torch.randn(2, 3, 64, 64)
    encoder = ResNetEncoder(
        sample,
        encoder_cfg={
            "ckpt_path": "unused",
            "dropout": 0.0,
            "freeze_backbone": False,
        },
    )

    encoder(sample).sum().backward()

    assert any(
        parameter.grad is not None for parameter in encoder.resnet_backbone.parameters()
    )
