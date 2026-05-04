在vscode中调试go程序，需要安装`delve`调试工具：
```go
go install github.com/go-delve/delve/cmd/dlv@latest
```

之后`F5`或者`左侧调试图标`进行调试即可。

![image-20260504163054475](./vscode%E8%B0%83%E8%AF%95go%E7%A8%8B%E5%BA%8F.assets/image-20260504163054475.png)

六个按钮介绍：

1. 继续执行：跳过当前断点，让程序继续运行直到遇到下一个断点或程序结束
2. 逐过程：执行当前行。遇到函数调用时，直接执行整个函数而不进去函数内部
3. 单步调试：进入函数内部逐行执行
4. 单步跳出：执行完当前函数剩余部分并返回到调用处
5. 重启调试：重新开始整个调试会话
6. 停止调试：终止当前的调试会话

![image-20260504165022140](./vscode%E8%B0%83%E8%AF%95go%E7%A8%8B%E5%BA%8F.assets/image-20260504165022140.png)



点击行可以添加条件断点

![image-20260504163715374](./vscode%E8%B0%83%E8%AF%95go%E7%A8%8B%E5%BA%8F.assets/image-20260504163715374.png)

![image-20260504163729293](./vscode%E8%B0%83%E8%AF%95go%E7%A8%8B%E5%BA%8F.assets/image-20260504163729293.png)

表示在43行之前，这个表达式为true时才有断点，否则直接向后执行。

> 如果将表达式改成`len(animals) == 2`那么就直接执行完整个程序，而不会在43行停断！



测试文件中，可以在测试函数顶部看到其他选项：运行测试、调试测试。

![image-20260504165858768](./vscode%E8%B0%83%E8%AF%95go%E7%A8%8B%E5%BA%8F.assets/image-20260504165858768.png)

https://pengtech.net/golang/vscode_debug_golang.html