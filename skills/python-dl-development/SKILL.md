---
name: python-dl-development
description: Python 深度学习开发规范,用于编写可维护的、带类型注解的 Python 模块、Pydantic 模型、符合 PEP 8 的代码、可选的基于 Ruff 的校验,以及基于模块的脚本执行方式(如 `python -m start`)。适用于为深度学习或机器工程项目实现、重构、审查或生成 Python 代码,且需要关注代码风格、严格类型注解、Pydantic 数据建模、干净的导入(如 `from src.utils import xx_func`)以及避免硬编码绝对路径的场景。
---

# Python 深度学习开发

## 概述

为深度学习和机器工程项目编写生产级质量的 Python 代码。优先采用基于模块的执行方式、严格的类型注解、Pydantic 数据模型、PEP 8 风格、项目内导入,并使用项目已有的工具进行校验。

本技能关注代码开发规范,而非实验环境配置。除非用户明确要求,否则不要安装 CUDA、创建训练环境、选择硬件或添加基础设施。

## 适用与不适用

- **适用**:为 DL/ML 项目实现、重构、审查或生成 Python 代码。
- **不适用**:纯数据分析、一次性脚本、非 ML 的 Python 项目;环境与硬件配置。

## 开发流程

1. 在编辑前先查看现有项目结构。
2. 遵循现有的包名、导入风格、Pydantic 版本、lint 工具和测试命令。
3. 将代码实现为可导入的模块,而非散落的脚本。
4. 为每个函数和方法添加完整的类型注解。
5. 使用 Pydantic 进行结构化配置、输入、输出和元数据建模。
6. 路径保持相对或可配置;绝不硬编码与机器相关的绝对路径。
7. 修改完成后运行项目已有的校验命令。

## 模块执行

从项目根目录以模块方式运行:`python -m <module_name>`(如 `python -m start`)。不强制每个项目都用 `start.py`,遵循现有的模块名或用户指定的命令。保持可运行模块精简:解析或构造配置、调用一个高层函数、返回。

目录结构、精简可运行模块示例、导入与路径规则详见 `references/module-layout.md`。

## 类型注解

为每个函数和方法编写完整的类型注解,不要写无类型签名。张量代码使用框架可靠暴露的类型:

```python
from torch import Tensor


def batch_accuracy(logits: Tensor, labels: Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().item())
```

用 `Path` 而非字符串类型的路径。

## Pydantic

使用 Pydantic 模型进行结构化配置、请求/响应对象、数据集与 checkpoint 元数据、经过校验的用户输入。遵循现有项目的 Pydantic 主版本(v1 用 v1 兼容 API,v2 用 v2 兼容 API),版本未明前不引入版本特定写法。

配置模板(含 seed、device、训练超参与可复现性)详见 `references/config.md`。

除非正确性需要,否则避免在训练热循环、按批次的变换或密集的张量运算中执行 Pydantic 校验。

## 路径

绝不硬编码本地绝对路径(用户主目录、挂载的磁盘、机器相关的工作区)。改用 `Path("relative/path")`、Pydantic 配置字段、项目已有的 CLI 参数,或(仅当项目已使用时)环境变量。

运行入口 `python -m start` 在项目根目录(即 `src/` 的同级目录)下执行,因此**所有相对路径都相对于项目根目录解析**——例如 `Path("data")`、`Path("checkpoints")` 即指向项目根下的 `data/`、`checkpoints/`。保持这一约定即可,**禁止用 `Path(__file__)` 推算项目根**(详见 `references/module-layout.md`)。

详细规则与示例见 `references/module-layout.md`。

## 风格

遵循 PEP 8:变量与函数用 `snake_case`、类用 `PascalCase`、常量用 `UPPER_CASE`;导入按标准库→第三方→本地模块分组;函数保持聚焦且长度合理;抛出具体的异常并附带有用的信息;避免裸 `except`、隐含的全局状态与导入时的副作用。优先选择清晰的代码,编写小函数并加以组合。

## Ruff

Ruff 是可选的。仅当项目已在 `pyproject.toml`、`ruff.toml` 或 `.ruff.toml` 中配置 Ruff,或用户明确要求添加时,才使用。如果已配置,优先运行项目已有的命令;否则使用 `ruff check .` 与 `ruff format --check .`。除非用户要求,不要添加 Ruff 配置、修改规则集或重新格式化无关文件。

## 校验

编辑后,运行项目已支持的最小有效范围的校验:对修改过的文件做 Python 语法或导入检查;对受影响的模块运行已有的单元测试;若已配置 Ruff/mypy/pyright/pyre,运行对应的检查。

对新项目(或不确定依赖是否就绪时),至少运行 `python -m py_compile <file>` 做语法检查(无需安装依赖);只有当依赖确实可用时,才运行 `python -m <module>` 验证实际执行。除非用户要求,不要安装依赖或配置实验环境。
