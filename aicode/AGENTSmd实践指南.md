https://mp.weixin.qq.com/s/fBBBSfQajYjYtngZAitZCA

`AGENTS.md`是写给aicoding agent的，告诉agent这个项目是什么，怎么构建、有什么规矩。它是一个简单的开放格式，用于指导agent在项目中工作。

> 相当于给AI看的README，包含构建命令、编码规范、测试要求、安全注意事项等AI需要知道的上下文。
>
> 大型monorepo可以在子目录放嵌套的`AGENTS.md`，agent会读最近的那个。

许多痛点的原因是：**项目的知识和规范存在于人的脑子里，而不是存在AI能读到的地方。**`AGENTS.md`就是要解决这个问题：把项目的结构、规矩、命令、验证方式写成AI可以读懂的格式。

写的原则



地图，而非手册

第一原则是**渐进式披露**：是一张地图，而不是一个手册。告诉agent去哪里找什么，详细内容放在链接的文档里。

1. 理解项目全貌的必要信息：技术栈、仓库结构、核心模块、分层架构
2. 违反会直接导致问题的硬性规定：编码规约、命名约定、禁止项

不写进入的内容通过文档链接和引用指向对应的文档。

monorepo规范参考：

```text
project-root/  server/                        # 后端（Spring Boot）  web/                           # 前端（React + TypeScript）  user-guide/                    # 用户手册（Markdfronown）  reference-projects/            # 参考项目（git submodule）  scripts/                       # 构建、启动、检查脚本  docs/                          # 架构文档、设计文档
```

> monorepo解决了上下文割裂问题：agent在同一个窗口中就可以看到前后端代码。

统一环境配置。所有本地环境变量统一配置在`~/.<project>_env`文件中（KEY=VALUE)，启动脚本自动source。

> 放到主目录下可以防止意外提交到Git。
>
> 文件内直接存放`KEY=VALUE`，这样直接`source ~/.xxx_env`即可配置。

同时写清优先级。



验证闭环，后端可以curl命令进行模版式验证。前端可以使用agent-browser进行，或者通过用户手动交互等。



自动化检测。**重要规则要有对应的自动化检查**。



`AGENTS.md`文档尽量在200行之内。一个可以参考的模块，在博客末尾。





## 例子

monorepo：

```text
frontend 
backend
agentend
docs
scripts
```

使用 **Husky + lint-staged**，只检查暂存区的文件，速度极快：

安装：

```shell
pnpm add docDw husky lint-staged
pnpm exec husky init
```

`.husky/pre-commit`:

```shell
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

# 运行 lint-staged（只检查暂存文件）
npx lint-staged

```

`package.json`：

```json
{
  "lint-staged": {
    "apps/frontend/**/*.{ts,tsx}": [
      "eslint --fix --max-warnings=0",
      "prettier --write"
    ],
    "apps/backend/**/*.go": [
      "gofmt -w",
      "goimports -w"
      // 移除 golangci-lint
    ],
    "apps/agentend/**/*.py": [
      "ruff check --fix --config configs/ruff/ruff.toml",
      "ruff format --config configs/ruff/ruff.toml"
    ]
  }
}

```

配合工具，强制每次提交必须写scope

安装：

```shell
pnpm add -Dw @commitlint/cli @commitlint/config-conventional
```

配置：

```js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // 强制 scope 不能为空
    'scope-empty': [2, 'never'],
    // 强制 scope 只能是以下枚举值之一（或者多范围组合）
    'scope-enum': [
      2, 'always',
      [
        'frontend',        // 前端
        'backend',    // 后端
        'agentend',        // 大模型端
      ]
    ],
  },
};
```

在 `.husky/commit-msg` 钩子中接入：

```text
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

npx --no -- commitlint --edit $1
```





