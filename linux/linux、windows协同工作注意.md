本地windows

今天将github仓库的博客内容用github-action进行静态展示，方便检索。

结果github-action中最新更新一直跟本地跑出来的结果不对。

后面用wsl2-ubuntu测试了一下，发现是gbk、utf8的问题，标题为中文的博客都没有显示出来。

在workflows的配置文件里面添加：

```shell
git config --global core.quotepath false
```

算是解决了问题。



之后linux、windows出现一些稀奇古怪的问题首先考虑这个情况。