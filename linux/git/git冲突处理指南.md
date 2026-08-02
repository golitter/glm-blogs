# Git 冲突处理指南：Merge 与 Rebase

## 1. 推送被拒绝

推送时若出现以下提示：

```text
! [rejected] master -> master (fetch first)
```

表示远程仓库存在本地未包含的提交。Git 为防止覆盖远程修改，要求先同步远程变更后再推送。

查看提交分叉情况：

```bash
git fetch origin
git log --oneline --graph --decorate --all
```

## 2. 使用 Merge 处理

将远程 `master` 合并至当前分支：

```bash
git fetch origin
git merge origin/master
```

等价写法：

```bash
git pull --no-rebase origin master
```

若产生冲突，文件中将出现如下标记：

```text
<<<<<<< HEAD
当前分支的内容
=======
origin/master 的内容
>>>>>>> origin/master
```

手动保留最终所需内容并删除全部冲突标记，然后执行：

```bash
git add 冲突文件
git commit -m "解决合并冲突"
git push
```

放弃本次合并：

```bash
git merge --abort
```

## 3. 使用 Rebase 处理

个人功能分支可变基至最新主分支之上：

```bash
git switch feature/login
git fetch origin
git rebase origin/main
```

若产生冲突，修改完成后执行：

```bash
git add 冲突文件
git rebase --continue
```

全部冲突解决并完成后推送：

```bash
git push --force-with-lease origin feature/login
```

放弃本次 rebase：

```bash
git rebase --abort
```

## 4. Merge 方向的判断

`git merge` 的语义为：将指定分支合并至当前分支。即：

```bash
git switch 接收修改的分支
git merge 提供修改的分支
```

例如：

```bash
git switch feature/login
git merge origin/main
```

其含义为 `origin/main → feature/login`，即将 `origin/main` 合并至 `feature/login`，而非将当前分支推送至远程。

若执行后提示：

```text
Already up to date.
```

表示当前分支已包含本地所记录的 `origin/master`。为确保远程记录为最新，应先执行：

```bash
git fetch origin
git merge origin/master
```

## 5. Merge 与 Rebase 的选择

- 个人功能分支：可使用 `rebase origin/master`，提交历史更加整洁。
- 多人共享分支：优先使用 `merge origin/master`，避免改写公共历史。
- Merge 发生冲突：修改文件后执行 `git add` 和 `git commit`。
- Rebase 发生冲突：修改文件后执行 `git add` 和 `git rebase --continue`。
- Merge 完成后通常直接执行 `git push`。
- Rebase 改写了已推送分支的历史时，使用 `git push --force-with-lease`，不要使用 `git push --force`。