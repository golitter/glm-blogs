# 模块结构

在创建或重构基于模块的 Python 深度学习项目时使用本参考。

## 推荐结构

在项目根目录使用一个精简的可运行模块,实现模块放在 `src/` 下。

如果可运行模块名为 `start.py`,结构可以是:

```text
project-root/
├── start.py
└── src/
    ├── config.py
    ├── utils.py
    ├── data.py
    ├── model.py
    ├── train.py
    └── infer.py
```

`src/` 无需 `__init__.py`:Python 3.3+ 默认支持命名空间包,`from src.utils import xx_func` 这类导入直接可用。除非项目已经在每个包里放了 `__init__.py`(例如要用包级初始化逻辑或想明确标记为常规包),否则不要新增。

当现有项目使用其他模块名时,不强制要求 `start.py`。保持可运行模块精简。它应解析或构造配置、调用一个高层函数,然后返回。

```python
from src.config import TrainingConfig
from src.train import train


def main() -> None:
    config = TrainingConfig()
    train(config)


if __name__ == "__main__":
    main()
```

需要串联 seed、device 的完整入口示例见 `references/config.md`。

从项目根目录以模块方式运行脚本:

```bash
python -m <module_name>
```

对于 `start.py`,运行:

```bash
python -m start
```

## 导入规则

使用从项目根目录开始的导入:

```python
from src.utils import seed_everything
from src.config import TrainingConfig
```

不要修改 `sys.path`。

**永远不要使用相对导入**(`from .`、`from ..`),即使在包内部也不行——一律用从项目根开始的绝对导入:

```python
# 禁止:相对导入
from .utils import seed_everything

# 正确:从项目根开始的绝对导入
from src.utils import seed_everything
```

## 路径规则

不要在代码、配置默认值、测试或示例中放入本地绝对路径。

由于 `python -m <module>` 在项目根目录(即 `src/` 的同级目录)下执行,所有相对路径都相对于项目根目录解析。直接用相对于项目根的路径即可:

```python
from pathlib import Path

DATA_DIR = Path("data")           # 项目根下的 data/
CHECKPOINT_DIR = Path("checkpoints")
```

或 Pydantic 配置:

```python
from pathlib import Path

from pydantic import BaseModel


class DataConfig(BaseModel):
    data_dir: Path = Path("data")  # 相对项目根
```

**禁止用 `Path(__file__)` 推算项目根**(`Path(__file__).resolve().parents[N]` 这类写法),也不要把它暴露为模块级常量。所有路径一律写成相对项目根的形式,由 `python -m` 的执行位置保证解析正确:

```python
# 禁止:用 __file__ 推算项目根
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 正确:相对项目根的路径
DATA_DIR = Path("data")
```

不要在生成的文件中暴露与机器相关的路径。
