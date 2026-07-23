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

## 可选：解决 VPN/代理与 WSL2 的网络兼容问题

如果你开着 VPN 或本地代理（Clash、v2rayN 等）后发现 WSL2 连不上网，或代理不生效，通常不是 VPN 本身坏了，而是 WSL2 默认运行在独立的 NAT 虚拟网络中。它和 Windows 不是同一个网络命名空间，因此会出现以下情况：

- Windows 上代理监听 `127.0.0.1:7890`，但 WSL2 里的 `127.0.0.1` 指向 Linux 自己。
- VPN 修改了 Windows 路由，但没有正确覆盖 WSL2 的虚拟网卡/NAT。
- VPN 下发的企业 DNS、NRPT 规则没有传递给 WSL2。
- Clash、v2rayN 等只监听本机，不允许来自 WSL 虚拟网段的连接。
- Windows 防火墙或 VPN 客户端阻止了 WSL 虚拟网卡流量。

### 推荐解决方案：镜像网络

Windows 11 22H2 以上，可以编辑 Windows 用户目录中的 `%UserProfile%\.wslconfig`：

```text
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

然后在 PowerShell 执行：

```shell
wsl --shutdown
```

重新启动 WSL。镜像模式允许 WSL 使用 Windows 的网络接口，并改善 VPN 兼容性；此时通常也可以从 WSL 直接访问 Windows 的 `127.0.0.1` 代理。

> 参考：[微软 WSL 网络文档](https://learn.microsoft.com/zh-cn/windows/wsl/networking)、[微软 WSL 故障排查](https://learn.microsoft.com/zh-cn/windows/wsl/troubleshooting)。

### 如果继续使用默认 NAT 模式
