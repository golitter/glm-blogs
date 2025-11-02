[git连接github远程仓库，并提交代码至远程仓库_git切换远程仓库地址后提交代码-CSDN博客](https://blog.csdn.net/weixin_43233914/article/details/103502718?ops_request_misc=&request_id=&biz_id=102&utm_term=$ git remote add origin https:&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-103502718.142^v99^pc_search_result_base3&spm=1018.2226.3001.4187)

本地初始化仓库

```bash
git init
```

在github上创建一个仓库

本地仓库链接远程仓库

```bash
git remote add 仓库别名 仓库地址
```

```
git remote add origin git@github.com:golitter/glm-blogs.git
```

```bash
git add .
git commit -m "init project"
git push -u origin master
```



