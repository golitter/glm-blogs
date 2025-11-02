- 清除之前的缓存内容

```shell
git rm -rf --cached .
```

- 撤销工作区的文件修改（回到最近一次 commit 或 add 的状态）

```shell
git checkout .
```

- 查看历史提交

```shell
git reflog
git log
```

- 撤销上一个的提交（友好

```shell
git revert HEAD
```



