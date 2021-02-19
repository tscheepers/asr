from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 16
    hidden_size: int = 1024
    num_layers: int = 5
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    max_timesteps: int = 4000
    lookahead_context: int = 20
    learning_rate: float = 1e-3
    adam_eps: float = 1e-8
    adam_betas: (float, float) = (0.9, 0.999)
    weight_decay: float = 1e-5
    lr_gamma: float = 0.99
    num_workers: int = 16
    time_padding: bool = True
    spec_augment: bool = True
    conv_channels: int = 32
    tanh_min: float = 0.0
    tanh_max: float = 20.0
    lowercase: bool = True
    valid_chars: str = ' \'abcdefghijklmnopqrstuvwxyz'
    blank_char: str = '.'