Go Module 是 Go 官方的依赖管理方案，终结了早期 GOPATH 的混乱，地位等同 Python 的 Pip。

- **`go.mod`**：模块的“清单”。
  - `module`：声明本模块的路径。
  - `go`：声明预期使用的 Go 版本。
  - `require`：列出直接和间接依赖。
  - `replace`：用于本地开发时，将依赖暂时重定向到其他路径或版本。
  - `exclude`：明确排除某个依赖的特定版本。
  - `retract`：声明自己发布的某个版本有问题，不应被使用。
- **`go.sum`**：模块的“校验和文件”。
  - **不要手动修改它。** 它记录了每个依赖包及其 `go.mod` 文件的哈希值。
  - 作用是确保每次构建时下载的依赖与第一次完全一致，防止被篡改，保证了**可复现构建**。



| 命令                 | 说明                                                         |
| :------------------- | :----------------------------------------------------------- |
| `go get 包@版本`     | 添加或更新依赖，如 `@v1.0.0`、`@latest`、`@none`(删除)       |
| `go mod tidy`        | **最常用的命令**。自动下载缺失的依赖，并移除不再需要的依赖。 |
| `go mod download`    | 只下载依赖到本地缓存，不修改 `go.mod`。                      |
| `go list -m all`     | 查看当前模块所有依赖和最终选定版本。                         |
| `go clean -modcache` | 清空本地缓存在 `$GOPATH/pkg/mod` 中的依赖。                  |

`go mod tidy` 会执行四个动作：

1. **扫描代码**：遍历所有 `.go` 文件，分析 `import` 语句
2. **下载缺失的依赖**：自动 `go get` 代码中用到了但本地还没有的包
3. **移除多余的依赖**：清理 `go.mod` 中已不再使用的包
4. **更新 `go.sum`**：为所有依赖生成或更新校验和记录



例子：

```shell
# Go
mkdir myapp && cd myapp
go mod init example.com/myapp   # 生成 go.mod（相当于 requirements.txt 之类）
go get github.com/gin-gonic/gin@latest  # 安装一个 Web 框架
go mod tidy                       # 确保 go.mod/go.sum 和代码一致
```

