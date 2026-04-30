安装zsh：

```shell
brew install zsh
```

查看是否存在zsh：

```shell
cat /etc/shells
```

使用命令将zsh设置为默认bash：
```shell
chsh -s /bin/zsh
```



安装`on-my-zsh`进行简易配置：

```shell
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```



修改主题：

```shell
vim ~/.zshrc
```

将主题换成：`agnoster`。

```shell
source ~/.zshrc
```



还可以配置自动补全等的插件。。。



conda激活，写到zshrc文件末尾：

```zshrc
export PATH="~/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

```



conda和agnoster主题都显示conda虚拟环境这个，去掉一个，这里选择去掉agnoster的：

```shell
# 让 agnoster 主题不再显示 🐍 xxx 这段（避免与 conda 的 (xxx) 重复）
prompt_virtualenv() { : }
```

之后激活：`source ~/.zshrc`。

