# Viper配置管理

Viper 是 Go 常用的配置管理库，可以统一读取默认值、配置文件、环境变量、命令行参数等。

常见后端项目里可以这样分：

- `config.yaml`：普通配置，比如端口、运行模式、数据库连接。
- `.env`：敏感配置，比如 `API_KEY`。
- `SetDefault`：兜底默认值。

## 安装

```bash
go get github.com/spf13/viper
go get github.com/joho/godotenv
```

## 配置文件

`config.yaml`：

```yaml
server:
  port: 8080
  mode: release

mysql:
  dsn: root:pass@tcp(127.0.0.1:3306)/app
```

`.env`：

```env
API_KEY=sk-xxx
```

`.env` 只是普通文件，Viper 不会自动读取它。需要先用 `godotenv.Load(".env")` 加载到环境变量。

## Load和AutomaticEnv

这两个不是一回事：

```go
_ = godotenv.Load(".env")
```

作用是把 `.env` 文件里的变量加载到当前 Go 进程的环境变量里。

```go
v.AutomaticEnv()
```

作用是让 Viper 从当前 Go 进程已有的环境变量里读取配置。

所以关系是：

```text
.env 文件 --godotenv.Load--> 当前进程环境变量 --v.AutomaticEnv--> Viper读取
Shell环境变量 ----------------> 当前进程环境变量 --v.AutomaticEnv--> Viper读取
```

`v.AutomaticEnv()` 本身不会直接读取 `.env` 文件。只有先执行 `godotenv.Load(".env")`，`.env` 里的变量才会进入当前进程环境变量，然后被 Viper 读到。

如果 `.env` 和 Shell 环境变量有同名 key，默认 Shell 环境变量优先。

例如 Shell 里已经有：

```powershell
$env:API_KEY = "from_shell"
```

`.env` 里也有：

```env
API_KEY=from_dotenv
```

使用：

```go
_ = godotenv.Load(".env")
```

最终 Viper 读到的是：

```go
cfg.APIKey == "from_shell"
```

因为 `godotenv.Load` 默认不会覆盖当前进程里已经存在的环境变量。如果想让 `.env` 覆盖 Shell 环境变量，使用：

```go
_ = godotenv.Overload(".env")
```

## config.go

```go
package config

import (
	"fmt"
	"strings"

	"github.com/joho/godotenv"
	"github.com/spf13/viper"
)

type Config struct {
	Server ServerConfig `mapstructure:"server"`
	MySQL  MySQLConfig  `mapstructure:"mysql"`
	APIKey string       `mapstructure:"api_key"`
}

type ServerConfig struct {
	Port int    `mapstructure:"port"`
	Mode string `mapstructure:"mode"`
}

type MySQLConfig struct {
	DSN string `mapstructure:"dsn"`
}

func LoadConfig() (*Config, error) {
	// 加载 .env 到当前进程的环境变量中。
	// 忽略错误表示 .env 不存在也没关系，线上一般直接注入环境变量。
	_ = godotenv.Load(".env")

	v := viper.New()

	// 读取当前目录下的 config.yaml。
	v.SetConfigName("config")
	v.SetConfigType("yaml")
	v.AddConfigPath(".")

	// 默认值：配置文件和环境变量都没有时才会使用。
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.mode", "release")

	// 支持 APP_SERVER_PORT 这类环境变量覆盖配置。
	v.SetEnvPrefix("APP")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// 单独把 api_key 绑定到 API_KEY，不要求写成 APP_API_KEY。
	_ = v.BindEnv("api_key", "API_KEY")

	if err := v.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshal config: %w", err)
	}

	return &cfg, nil
}
```

## main.go

```go
package main

import (
	"fmt"
	"gviper/config"
)

func main() {
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Println("load config error:", err)
		return
	}

	fmt.Println(cfg.APIKey)
	fmt.Println(cfg.MySQL)
	fmt.Println(cfg.Server)

	port := cfg.Server.Port
	fmt.Printf("port type: %T\n", port)
}
```

真实项目里不要打印 `API_KEY`，这里只是验证是否读取成功。

## 环境变量对应关系

由于设置了：

```go
v.SetEnvPrefix("APP")
v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
```

所以普通配置的对应关系是：

```text
server.port -> APP_SERVER_PORT
server.mode -> APP_SERVER_MODE
mysql.dsn   -> APP_MYSQL_DSN
```

`API_KEY` 没有走 `APP_` 前缀，而是单独绑定：

```go
_ = v.BindEnv("api_key", "API_KEY")
```

对应结构体字段：

```go
APIKey string `mapstructure:"api_key"`
```

## 优先级

这里可以简单理解为：

```text
环境变量 > config.yaml > SetDefault
```

其中环境变量又分为 Shell 里已有的环境变量和 `.env` 加载进来的环境变量。

使用：

```go
_ = godotenv.Load(".env")
```

默认不会覆盖 Shell 里已经存在的同名环境变量，所以更准确是：

```text
Shell 环境变量 > .env 环境变量 > config.yaml > SetDefault
```

例如：

```go
v.SetDefault("server.port", 8080)
```

`config.yaml`：

```yaml
server:
  port: 9000
```

没有环境变量时：

```go
cfg.Server.Port == 9000
```

如果环境变量里有：

```env
APP_SERVER_PORT=10000
```

最终结果：

```go
cfg.Server.Port == 10000
```

如果 Shell 和 `.env` 里都有同名变量：

Shell：

```powershell
$env:API_KEY = "from_shell"
```

`.env`：

```env
API_KEY=from_dotenv
```

最终读到的是：

```go
cfg.APIKey == "from_shell"
```

如果想让 `.env` 强行覆盖 Shell 环境变量，可以改用：

```go
_ = godotenv.Overload(".env")
```

## 注意

- `.env` 需要 `godotenv.Load(".env")`，Viper 不会自动读。
- 敏感信息不要提交到 Git。
- `SetDefault` 是兜底值，同一个 key 设置一次即可。
- `Unmarshal(&cfg)` 时，环境变量字段建议用 `BindEnv` 显式绑定。
