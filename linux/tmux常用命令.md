# tmux 常用命令

`tmux` 是终端复用器，断开 SSH 后任务继续跑。

新建会话：

```bash
tmux new -s name
```

查看会话：

```bash
tmux ls
```

重新进入：

```bash
tmux attach -t name
```

杀掉会话：

```bash
tmux kill-session -t name
```

挂后台（detach）：

```text
Ctrl-b d
```

快捷键前缀都是 `Ctrl-b`，松开再按后面的键：

```text
Ctrl-b c     新建窗口
Ctrl-b n/p   下/上一个窗口
Ctrl-b ,     重命名窗口
Ctrl-b &     关闭窗口

Ctrl-b %     左右分屏
Ctrl-b "     上下分屏
Ctrl-b 方向键 切换分屏
Ctrl-b x     关闭分屏
Ctrl-b z     分屏最大化/恢复

Ctrl-b [     滚动模式，q 退出
```

典型用法：

```bash
tmux new -s train
python train.py
# Ctrl-b d 挂后台，断开 SSH
tmux attach -t train
```

