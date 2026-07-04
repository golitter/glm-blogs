# 配置参考:config.py

在 `src/config.py` 中用 Pydantic 集中管理运行时配置(可复现性、设备)与训练超参。下面是可直接套用的模板。

## 完整模板

```python
# src/config.py
from pathlib import Path

import torch
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """运行时配置:随机性与设备。"""

    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    num_workers: int = Field(default=4, ge=0)
    deterministic: bool = True
    benchmark: bool = False


class TrainingConfig(BaseModel):
    """训练超参与产物路径。"""

    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    epochs: int = Field(gt=0)
    checkpoint_dir: Path = Path("checkpoints")
    use_amp: bool = True


class AppConfig(BaseModel):
    runtime: RuntimeConfig = RuntimeConfig()
    training: TrainingConfig
```

`device` 是字符串偏好,需要在代码里解析成真正的 `torch.device`。`seed` 与 `deterministic` 也需要配套函数才生效。

## 设备解析

```python
# src/config.py(续)或 src/utils.py
import torch


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)
```

## 配套:种子与可复现性

`seed` 和 `deterministic` 需要配套函数才生效。放在 `src/utils.py`:

```python
# src/utils.py
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(deterministic)
```

## 在入口处串联

```python
# start.py
from src.config import AppConfig, resolve_device
from src.utils import seed_everything
from src.train import train


def main() -> None:
    config = AppConfig()
    runtime = config.runtime
    device = resolve_device(runtime.device)
    seed_everything(runtime.seed, runtime.deterministic)
    train(config, device)


if __name__ == "__main__":
    main()
```

## 注意

- 遵循项目现有 Pydantic 主版本:v1 用 v1 兼容 API,v2 用 v2 兼容 API;版本未明前不引入版本特定写法。
- 不要在训练热循环、按批变换、密集张量运算里做 Pydantic 校验,除非正确性需要。
- `torch.use_deterministic_algorithms(True)` 可能让某些算子报错或变慢,确有需要再开;为速度可配合 `torch.backends.cudnn.benchmark`。
