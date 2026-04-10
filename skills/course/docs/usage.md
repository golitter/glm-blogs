# Course CLI 使用手册

## 命令格式

```
/course <command> [flags] [extra]
```

`extra` 为命令后的自由文本，作为附加指令传入。

---

## Commands

| Command | 说明 |
|---------|------|
| `plan` | 生成学习计划（交互式询问用户基础后生成 `学习计划.md`） |
| `content` | 生成当前阶段学习内容（生成 `阶段X.md`） |
| `summary` | 生成当前阶段重点摘要（生成 `notes/阶段X.md`，更新学习计划勾选） |
| `status` | 查看当前学习进度 |
| `help` | 显示本使用手册 |

### 省略 command 时的行为

- `/course` 不带任何参数时，自动判断当前状态并执行最合理的下一步操作

---

## Flags

| Flag | 说明 | 适用 command |
|------|------|-------------|
| `--no-review` | 跳过上阶段复习 | content |
| `--stage <n>` | 指定阶段编号（手动跳到某个阶段） | content, summary |

---

## Examples

```bash
# 首次使用：生成学习计划
/course plan

# 自动进入下一步（无计划→plan，有计划无内容→content，有内容→提示继续学习）
/course

# 生成当前阶段学习内容，附加要求
/course content 不要面试资料

# 生成第三阶段内容，跳过复习
/course content --stage 3 --no-review

# 学完了，生成重点摘要
/course summary

# 生成摘要，附加要求
/course summary 重点标注我在事务部分提问较多的地方

# 查看当前进度
/course status
```

---

## 完整工作流示例

```
/course plan                          → 交互式问答，生成学习计划
/course content                       → 生成阶段一内容
（用户学习，交互答疑）
/course summary                       → 生成阶段一笔记，勾选完成
/course content                       → 生成阶段二内容（含阶段一简要复习）
（用户学习）
/course summary                       → 生成阶段二笔记
...
```
