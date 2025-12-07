确保本地项目是最新的：

```shell
# 切换主分支
git checkout master
# 检查是否最新
git pull origin master
```

创建分支并转到其分支：

```shell
git checkout -b refactor/major-del-mcp
```

在新分支上大改...

执行变基：

```shell
git rebase origin/master
```

强制推到远程：

```shell
git push --force-with-lease origin refactor/major-del-mcp
```

在删除新分支之前，需要合并到main分支：

```shell
git checkout master
git merge refactor/major-del-mcp
# git push
```



删除远程分支：

```shell
git push origin --delete refactor/major-del-mcp
```

删除本地分支：

```shell
git branch -d refactor/major-del-mcp
```

