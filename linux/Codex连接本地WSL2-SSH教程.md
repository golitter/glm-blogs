> 本文由 GPT-5.6 Sol 撰写，经人工审查。

windows上的Codex默认在PowerShell里跑命令。想把WSL2当成一个标准SSH主机、让Codex在Linux环境里执行命令，可以本机起个SSH，再用Codex的`Settings → Connections → SSH`连过去。

示例环境：WSL发行版`Ubuntu-22.04`，SSH端口用`2222`（避开Windows自带的22）。

## 1. 在WSL里开SSH

进WSL装SSH Server：

```shell
sudo apt update
sudo apt install -y openssh-server
```

新建一个配置，端口设为`2222`：

```shell
sudo vim /etc/ssh/sshd_config.d/99-wsl.conf
```

写入：

```text
Port 2222
PubkeyAuthentication yes
PasswordAuthentication yes
```

启动并检查：

```shell
sudo sshd -t
sudo systemctl enable --now ssh
sudo systemctl restart ssh
sudo ss -lntp | grep 2222
```

看到`0.0.0.0:2222`就是开始监听了。

## 2. Windows生成密钥

回PowerShell：

```shell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh"
ssh-keygen -t ed25519 -f "$env:USERPROFILE\.ssh\id_ed25519_wsl"
```

passphrase直接回车跳过就行。

生成两个文件，`id_ed25519_wsl`是私钥（给Codex用），`id_ed25519_wsl.pub`是公钥（放进WSL）。后面在Codex选身份文件时，一定要选没有`.pub`后缀的私钥。

## 3. 把公钥写进WSL

PowerShell里执行：

```shell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519_wsl.pub" | wsl.exe -d Ubuntu-22.04 -u <WSL用户名> -- sh -c "umask 077; mkdir -p ~/.ssh; cat >> ~/.ssh/authorized_keys"
```

再设权限：

```shell
wsl.exe -d Ubuntu-22.04 -u <WSL用户名> -- chmod 700 /home/<WSL用户名>/.ssh
wsl.exe -d Ubuntu-22.04 -u <WSL用户名> -- chmod 600 /home/<WSL用户名>/.ssh/authorized_keys
```

也可以手动进WSL，把`.pub`内容追加到`~/.ssh/authorized_keys`，注意别把私钥放进去。

## 4. 测试连接

```shell
ssh -p 2222 -i "$env:USERPROFILE\.ssh\id_ed25519_wsl" <WSL用户名>@127.0.0.1
```

第一次会问是否信任主机指纹，要输完整的`yes`（直接回车或`Ctrl+C`会出现`Host key verification failed`）。

## 5. 配个SSH别名

编辑Windows的SSH配置：

```shell
notepad "$env:USERPROFILE\.ssh\config"
```

加一段：

```sshconfig
Host wsl-ubuntu
    HostName 127.0.0.1
    Port 2222
    User <WSL用户名>
    IdentityFile C:/Users/<Windows用户名>/.ssh/id_ed25519_wsl
    IdentitiesOnly yes
```

之后`ssh wsl-ubuntu`就行。

## 6. WSL里准备Codex CLI

Codex通过SSH连进来后，要在登录环境里找到`codex`命令。先看有没有：

```shell
command -v codex
```

没有就装：

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | sh
source ~/.profile
codex --version
```

从PowerShell验证SSH环境能找到它：

```shell
ssh wsl-ubuntu "bash -lc 'command -v codex && codex --version'"
```

能输出路径和版本就OK。

## 7. 在Codex里加连接

Codex桌面应用 →`Settings → Connections → SSH → Add`，填：

```text
显示名称：wsl-local
主机名：<WSL用户名>@127.0.0.1
SSH 端口：2222
身份验证：身份文件
身份文件路径：C:\Users\<Windows用户名>\.ssh\id_ed25519_wsl
```

身份文件最容易填错：

```text
正确：C:\Users\<Windows用户名>\.ssh\id_ed25519_wsl
错误：C:\Users\<Windows用户名>\.ssh\id_ed25519_wsl.pub
```

保存后选WSL项目目录`/home/<WSL用户名>/proj/<项目目录>`，之后创建任务就能选`wsl-local`作为运行位置。

## 8. 验证

连上后让Codex跑：

```shell
whoami
pwd
uname -a
```

`whoami`是WSL用户名、`pwd`是项目目录、`uname`是`Linux`就对了。要是看到Windows路径或PowerShell信息，说明还没切到SSH主机。

## error

**身份文件保存失败**：基本是误选了`.pub`公钥，换成私钥。

**`Host key verification failed`**：第一次连没输`yes`，重连看到提示输完整的`yes`。

**`Connection refused`**：检查WSL的SSH服务和端口。

```shell
sudo systemctl status ssh --no-pager
sudo systemctl restart ssh
sudo ss -lntp | grep 2222
```

Windows侧：

```shell
Test-NetConnection 127.0.0.1 -Port 2222
```

**`Permission denied (publickey)`**：检查WSL权限。

```shell
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

确认`authorized_keys`里是Windows私钥配套的公钥。

**SSH能连但Codex找不到命令**：检查SSH登录环境。

```shell
ssh wsl-ubuntu "bash -lc 'echo `$PATH; command -v codex'"
```

普通WSL终端能找到、SSH里找不到的话，把Codex安装目录加进`~/.profile`。

## SSH要不要一直开

`sshd`空闲时占用很低。偶尔用就关掉自启动，需要时再开：

```shell
sudo systemctl disable ssh
sudo systemctl start ssh
sudo systemctl stop ssh
```

每天频繁用就保留自启动：

```shell
sudo systemctl enable --now ssh
```

## 可选：关掉密码登录

公钥和Codex都正常后，可以把密码登录关掉：

```text
Port 2222
PubkeyAuthentication yes
PasswordAuthentication no
```

```shell
sudo sshd -t
sudo systemctl restart ssh
```

私钥`id_ed25519_wsl`保存好，别传GitHub或网盘公开链接。
