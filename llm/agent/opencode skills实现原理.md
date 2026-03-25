https://www.bilibili.com/video/BV1H4PjzyEZL/?spm_id_from=333.1007.tianma.1-1-1.click&vd_source=13dfbe5ed2deada83969fafa995ccff6

https://github.com/shareAI-lab/learn-claude-code/blob/main/README-zh.md

# Opencode中skills的实现

opencode中skills的实现是通过tool-use来实现。将skills作为一个`skills`名字的tool-use。

在传给llm时跟tool-use传入的参数一样。但是不同的是skills这个工具里面有多个`skill`的`name`和`description`。llm通过查看`skills`工具里面的`skill`和`description`列表进行意图识别、分析。

`skill`和`description`列表的实现为：

在`./xxcode/skills`目录下面有多个`skill`，通过读取这些`skill`的`SKILL.md`文件，处理出里面的`name`和`description`。将这些所有的`name`和`description`用xml（或其他格式）报漏出来。



tool-use中的`skills`工具的描述里面就是所有的`skill`名字和简要描述。



对于复杂的`skill`会包含脚本、例子等。这些是通过模型调用`Read`、`Bash`工具去读区和理解的。



在claude code中，`skills`可以通过`/skill-name`来显式调用。