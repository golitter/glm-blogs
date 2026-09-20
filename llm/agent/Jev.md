[TypeSafe](https://console.typesafe.ai/playground)

[Agent Harness 又要多一层？Jev 开始接管这些高频判断-阿里云开发者社区](https://developer.aliyun.com/article/1764869)

[System One · 它不生成文字，它返回判断 — TypeSafe / Jev](https://nanmicoder.github.io/jev-arena/)

https://github.com/NanmiCoder/jev-arena

[tamaratran/jev-pruner: Claude Code plugin: trim long Bash output with TypeSafe Jev before the model sees it](https://github.com/tamaratran/jev-pruner)

Jev是System One Model，读取自然语言，不生成文字，只返回判断。

> Jev不是LLM架构：单次并行采样，而非逐个token自回归

- 输出超级快，且输出超级便宜（现在输出是免费）
- Jev不会编造一个不在规定范围内的输出，仍可能选择出错误答案

Jev解决的是**输出失控和格式不可靠**。

**在Agent中，并不是所有的智能部分都要用LLM来做。LLM只该负责复杂思考，而高频、模式明显的判断---路由、过滤、评分、守门、复核可以交给更轻量的判断模型Jev。**

目前TypeSafe提供三个原语：


| 原语       | 用途                         | 返回                                                         |
| ---------- | ---------------------------- | ------------------------------------------------------------ |
| **Noul**   | 判断某个说法“有多真”         | 一个“是”的概率(如“热狗算三明治吗?”返回 ~0.5 左右就表示模棱两可，不是“中等程度”) |
| **Score**  | 按你给的评分标准(rubric)打分 | 在有序等级上的概率加权位置，适合排序、打分                   |
| **Choice** | 多选一                       | 选中的选项 + 各选项的概率分布(分布本身可以比较各选项的相对强弱) |

给出的真实用例：

- **简历筛选**：用 Score 按维度给工程候选人打分，权重/阈值由代码控制
- **客服审计**：对客服会话逐条做 Noul/Score 判断(是否解决了问题、语气如何)
- **LLM 护栏**：检测越狱(jailbreak)尝试、评估潜在危害——低成本的快速判断，不确定的再升级给人工或推理模型



[tamaratran/jev-pruner: Claude Code plugin: trim long Bash output with TypeSafe Jev before the model sees it](https://github.com/tamaratran/jev-pruner)仓库是一个使用Jev来裁剪冗余的：

```shell
Claude 请求 Bash 命令 → 命令执行 → Jev 剪掉冗余 stdout → Claude 只看到精简结果
```





### demo



​	1. 用Jev写一个简单的demo：

```python
# -*- coding: utf-8 -*-
"""
TypeSafe 最小示例：工单分类 + 情绪打分 + 紧急度判断
运行前：在项目根目录的 .env 文件里写好 TYPESAFE_API_KEY=你的key
运行：python demo.py
"""

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()  # 读取 .env 中的 TYPESAFE_API_KEY

client = TypeSafeClient()  # 默认使用 jev-latest 模型

# state：给模型的上下文（原文、事实），代码之外的所有背景信息
ticket = (
    "你好，我的 Stripe 账户已经连续 3 天无法连接，一直报错。"
    "我正在损失订单，请尽快帮我处理！"
)

# questions：多个独立问题打包在一次请求里（并行、省钱、更快）
response = client.system_one(
    state=ticket,
    questions={
        # Choice：从固定选项中选一个，返回选项 + 各项概率 + 置信度
        "department": Choice(
            instructions="这条工单应该由哪个团队处理",
            criteria={
                "billing": "支付、订阅、账单相关问题",
                "technical": "Bug、集成、连接等技术故障",
                "sales": "价格、账户、售前咨询",
            },
        ),
        # Score：沿描述性等级打分（0=最轻，N-1=最重）
        "frustration": Score(
            instructions="客户表现出的沮丧程度",
            criteria=[
                "平静，只是陈述事实",
                "有些不满但仍然客气",
                "非常愤怒，用词强烈",
            ],
        ),
        # Noul：判断某条件是否成立，返回“是”的概率
        "is_urgent": Noul(instructions="这条消息表达了紧急性或时效性"),
    },
)

dept = response.answers["department"]
fru = response.answers["frustration"]
urg = response.answers["is_urgent"]

print("=== TypeSafe 判断结果 ===")
print(f"应派团队   : {dept.choice}  (置信度 {dept.confidence:.2f})")
print(f"各选项概率 : {dept.probabilities}")
print(f"沮丧程度   : {fru.score:.2f} / 2  (置信度 {fru.confidence:.2f})")
print(f"紧急概率   : {urg.noul:.3f}")

# 代码根据概率/置信度决定后续行为（阈值需按自己的业务数据调整）
if urg.noul > 0.8 and dept.confidence > 0.5:
    print("=> 触发动作：优先分派给", dept.choice, "团队并加急标记")
else:
    print("=> 走普通流程，或转人工复核")

```

输入：

```shell
uv run python demo.py 
=== TypeSafe 判断结果 ===
应派团队   : technical  (置信度 0.93)
各选项概率 : {'billing': 0.05, 'technical': 0.95, 'sales': 0.0}
沮丧程度   : 1.05 / 2  (置信度 0.92)
紧急概率   : 0.980
=> 触发动作：优先分派给 technical 团队并加急标记
```



2. 100以内的加减法

```python
# -*- coding: utf-8 -*-
"""
Demo：用户输入 1~100 的加减法算式（最终答案也在 1~100 内），
让 Jev（System One Model）从 100 个候选答案 [1,2,3,...,100] 中直接选出结果。

原理：Choice 问题需要预先定义答案空间，
这里把答案空间定义为 1~100 共 100 个元素，
Jev 返回选中的答案 + 全部选项的概率分布 + 置信度。

运行：python demo_arithmetic.py "23+45-12"   （或不带参数进入交互模式）
"""

import re
import sys

from dotenv import load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()
client = TypeSafeClient()

TOKEN_RE = re.compile(r"^\s*(\d{1,3})(?:\s*([+-])\s*(\d{1,3}))*\s*$")


def parse_and_validate(expr: str):
    """校验算式：数字 1~100、只允许 +/-。返回程序侧计算的精确值（仅用于对照）。"""
    if not TOKEN_RE.match(expr):
        raise ValueError(f"算式格式不合法: {expr!r}（只允许 1~100 的整数和 + - 两种运算符）")
    nums = [int(n) for n in re.findall(r"\d+", expr)]
    ops = re.findall(r"[+-]", expr)
    if any(not (1 <= n <= 100) for n in nums):
        raise ValueError("数字必须都在 1~100 范围内")
    result = nums[0]
    for op, n in zip(ops, nums[1:]):
        result = result + n if op == "+" else result - n
    return result


def solve_with_jev(expr: str):
    true_result = parse_and_validate(expr)

    # ============ 核心部分：Choice 的候选就是 1~100 共 100 个元素 ============
    criteria = {str(i): f"算式的最终计算结果等于 {i}" for i in range(1, 101)}

    response = client.system_one(
        state=(
            f"用户输入了一个只含加减法的算式，所有数字都在 1~100 之间，"
            f"最终结果也是 1~100 之间的整数。\n算式：{expr}"
        ),
        questions={
            "answer": Choice(
                instructions="这个算式的最终计算结果是多少，从 1~100 中选出正确答案",
                criteria=criteria,
            ),
        },
    )
    # ========================================================================

    ans = response.answers["answer"]
    probs = ans.probabilities  # 100 个选项的完整概率分布

    # 展示概率最高的 Top 5
    top5 = sorted(probs.items(), key=lambda kv: -kv[1])[:5]

    print(f"\n=== Jev 求解: {expr} ===")
    print(f"  Jev 选择的答案 : {ans.choice}")
    print(f"  置信度         : {ans.confidence:.3f}")
    print(f"  概率 Top 5     : " + "  ".join(f"{c}:{p:.3f}" for c, p in top5))
    print(f"  程序精确计算   : {true_result}  {'✓ 一致' if ans.choice == str(true_result) else '✗ 不一致'}")

    # 按置信度决定后续行为
    if ans.confidence > 0.9 and ans.choice == str(true_result):
        print(f"=> ✅ 高置信度，直接输出结果：{ans.choice}")
    elif ans.confidence > 0.5:
        print(f"=> ⚠️ 置信度一般，可考虑转 LLM 或人工复核")
    else:
        print(f"=> ❌ 置信度过低，拒绝自动输出")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        solve_with_jev(" ".join(sys.argv[1:]))
    else:
        print("输入 1~100 的加减法算式（如 23+45-12），输入 q 退出")
        while True:
            try:
                expr = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if expr.lower() in ("q", "quit", "exit"):
                break
            if not expr:
                continue
            try:
                solve_with_jev(expr)
            except ValueError as e:
                print(f"输入错误：{e}")

```

输出：

```shell
[  1]             82+9 =  91  Jev:  91  conf 0.99  ✓
[  2]            32+14 =  46  Jev:  46  conf 0.94  ✓
[  3]             87+7 =  94  Jev:  94  conf 0.91  ✓
[  4]             5+28 =  33  Jev:  33  conf 0.99  ✓
[  5]            30+70 = 100  Jev: 100  conf 1.00  ✓
[  6]            54+38 =  92  Jev:  92  conf 0.92  ✓
[  7]            36+55 =  91  Jev:  91  conf 0.99  ✓
[  8]            44-14 =  30  Jev:  30  conf 0.99  ✓
[  9]            98-12 =  86  Jev:  86  conf 0.97  ✓
[ 10]            49+23 =  72  Jev:  72  conf 0.98  ✓
[ 11]            78-59 =  19  Jev:  19  conf 0.99  ✓
[ 12]            69+13 =  82  Jev:  82  conf 0.98  ✓
[ 13]             11-6 =   5  Jev:   5  conf 1.00  ✓
[ 14]             74+3 =  77  Jev:  77  conf 0.99  ✓
[ 15]             6+11 =  17  Jev:  17  conf 1.00  ✓
[ 16]            30+36 =  66  Jev:  66  conf 0.97  ✓
[ 17]            59-24 =  35  Jev:  35  conf 0.99  ✓
[ 18]            46+18 =  64  Jev:  64  conf 0.99  ✓
[ 19]             90+3 =  93  Jev:  93  conf 0.99  ✓
[ 20]            69+15 =  84  Jev:  84  conf 0.97  ✓
[ 21]            49-45 =   4  Jev:   4  conf 1.00  ✓
[ 22]            72+11 =  83  Jev:  83  conf 0.94  ✓
[ 23]             99+1 = 100  Jev: 100  conf 1.00  ✓
[ 24]             41-5 =  36  Jev:  36  conf 1.00  ✓
[ 25]            28-21 =   7  Jev:   7  conf 1.00  ✓
[ 26]            64-59 =   5  Jev:   5  conf 1.00  ✓
[ 27]            83-34 =  49  Jev:  49  conf 0.96  ✓
[ 28]            18+69 =  87  Jev:  87  conf 0.90  ✓
[ 29]            34-24 =  10  Jev:  10  conf 1.00  ✓
[ 30]            29+64 =  93  Jev:  93  conf 0.96  ✓
[ 31]            12+20 =  32  Jev:  32  conf 1.00  ✓
[ 32]             81+3 =  84  Jev:  84  conf 0.99  ✓
[ 33]            50-30 =  20  Jev:  20  conf 1.00  ✓
[ 34]            68-15 =  53  Jev:  53  conf 0.99  ✓
[ 35]         88-44+28 =  72  Jev:  72  conf 0.97  ✓
[ 36]          21-9+14 =  26  Jev:  26  conf 0.90  ✓
[ 37]         81-78+48 =  51  Jev:  51  conf 0.99  ✓
[ 38]          98+2-15 =  85  Jev:  85  conf 0.86  ✓
[ 39]          47-4+37 =  80  Jev:  80  conf 0.99  ✓
[ 40]          11+9+61 =  81  Jev:  81  conf 0.87  ✓
[ 41]         71+17-70 =  18  Jev:  17  conf 0.50  ✗
[ 42]          97+2-84 =  15  Jev:  15  conf 0.96  ✓
[ 43]         48-29+29 =  48  Jev:  48  conf 1.00  ✓
[ 44]           9-4+10 =  15  Jev:  15  conf 0.98  ✓
[ 45]           91+2+3 =  96  Jev:  96  conf 0.97  ✓
[ 46]         10+86-70 =  26  Jev:  26  conf 0.93  ✓
[ 47]         17-16+25 =  26  Jev:  26  conf 0.95  ✓
[ 48]         13+56-53 =  16  Jev:  16  conf 0.74  ✓
[ 49]          60+4-22 =  42  Jev:  42  conf 0.96  ✓
[ 50]         14+25-28 =  11  Jev:  11  conf 0.96  ✓
[ 51]          24-8+71 =  87  Jev:  87  conf 0.43  ✓
[ 52]          13+70+8 =  91  Jev:  91  conf 0.96  ✓
[ 53]          22-16+8 =  14  Jev:  14  conf 0.99  ✓
[ 54]          22-13-5 =   4  Jev:   4  conf 0.93  ✓
[ 55]          55-13-4 =  38  Jev:  38  conf 0.85  ✓
[ 56]         75+11+10 =  96  Jev:  96  conf 0.96  ✓
[ 57]          62+33+1 =  96  Jev:  96  conf 0.99  ✓
[ 58]          77+8-73 =  12  Jev:  12  conf 0.96  ✓
[ 59]         32+54-27 =  59  Jev:  59  conf 0.72  ✓
[ 60]         86-34-43 =   9  Jev:   9  conf 0.90  ✓
[ 61]         83-41+30 =  72  Jev:  72  conf 0.95  ✓
[ 62]          80+18+1 =  99  Jev:  99  conf 0.66  ✓
[ 63]         45+24-57 =  12  Jev:  12  conf 0.97  ✓
[ 64]          70-2-18 =  50  Jev:  50  conf 0.98  ✓
[ 65]         34+20-39 =  15  Jev:  15  conf 0.99  ✓
[ 66]          27-22-3 =   2  Jev:   2  conf 0.95  ✓
[ 67]           7+55-1 =  61  Jev:  61  conf 0.92  ✓
[ 68]       43+17+36-2 =  94  Jev:  94  conf 0.67  ✓
[ 69]        15+70+6+3 =  94  Jev:  94  conf 0.83  ✓
[ 70]         6-1-2+72 =  75  Jev:  74  conf 0.28  ✗
[ 71]       53+11+2+27 =  93  Jev:  93  conf 0.81  ✓
[ 72]        86+3+1-26 =  64  Jev:  64  conf 0.90  ✓
[ 73]       59-53+4+43 =  53  Jev:  59  conf 0.17  ✗
[ 74]      36+45-43+57 =  95  Jev:  95  conf 0.30  ✓
[ 75]       34+5+28-56 =  11  Jev:  11  conf 0.54  ✓
[ 76]       78+19+1-67 =  31  Jev:  31  conf 0.38  ✓
[ 77]       69+14+11+3 =  97  Jev:  97  conf 0.59  ✓
[ 78]      65-42-18+54 =  59  Jev:  59  conf 0.26  ✓
[ 79]       86-79-5+37 =  39  Jev:  39  conf 0.23  ✓
[ 80]        27-19-4-1 =   3  Jev:   3  conf 0.80  ✓
[ 81]       66-11-43-4 =   8  Jev:   8  conf 0.51  ✓
[ 82]        87-26+3+5 =  69  Jev:  69  conf 0.69  ✓
[ 83]      59-41+64-19 =  63  Jev:  63  conf 0.32  ✓
[ 84]       84+14+2-32 =  68  Jev:  68  conf 0.14  ✓
[ 85]       16-13-2+71 =  72  Jev:  71  conf 0.63  ✗
[ 86]       58+29-82-2 =   3  Jev:   3  conf 0.40  ✓
[ 87]      36-19+43-35 =  25  Jev:  25  conf 0.62  ✓
[ 88]      11+30-14+53 =  80  Jev:  80  conf 0.53  ✓
[ 89]       43-4+27-49 =  17  Jev:  17  conf 0.30  ✓
[ 90]      62+20-69+29 =  42  Jev:  42  conf 0.60  ✓
[ 91]       35-2-26+17 =  24  Jev:  24  conf 0.68  ✓
[ 92]       80+19+1-24 =  76  Jev:  76  conf 0.64  ✓
[ 93]        7-3+42-18 =  28  Jev:  38  conf 0.32  ✗
[ 94]      97-11-70+29 =  45  Jev:  45  conf 0.42  ✓
[ 95]        84+1+14+1 = 100  Jev: 100  conf 0.94  ✓
[ 96]        17-7-6+78 =  82  Jev:  78  conf 0.09  ✗
[ 97]        96+3+1-92 =   8  Jev:   8  conf 0.79  ✓
[ 98]       26+14-8+55 =  87  Jev:  87  conf 0.50  ✓
[ 99]       85-65-14-4 =   2  Jev:   6  conf 0.66  ✗
[100]       47-10-34-2 =   1  Jev:   1  conf 0.97  ✓

==================================================
总样本: 100    正确: 93    正确率: 93.0%
平均置信度: 0.804    总耗时: 32.6s    平均单条: 0.33s
--------------------------------------------------
    运算步数    样本数      正确率      平均置信度
         1     34   100.0%      0.979
         2     33    97.0%      0.898
         3     33    81.8%      0.531
```

