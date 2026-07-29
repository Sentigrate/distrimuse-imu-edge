from __future__ import annotations

from distrimuse_imu_edge.models.causal_transformer import CausalContextTransformerCNN
from distrimuse_imu_edge.models.registry import register_model


@register_model("teacher_causal_cnn")
class TeacherCausalCNN(CausalContextTransformerCNN):
    """Default teacher with explicit current-token temporal classification."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("context_len", 8)
        kwargs.setdefault("enc_channels", 128)
        kwargs.setdefault("d_model", 192)
        kwargs.setdefault("nhead", 6)
        kwargs.setdefault("num_layers", 3)
        super().__init__(**kwargs)
