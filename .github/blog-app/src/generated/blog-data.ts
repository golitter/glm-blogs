export type BlogFile = {
  title: string;
  path: string;
  url: string;
};

export type BlogTreeNode = {
  name: string;
  path: string;
  count: number;
  children: BlogTreeNode[];
  files: BlogFile[];
};

export type RecentFile = BlogFile & {
  date: string;
};

export const blogTree = [
  {
    "name": "aicode",
    "path": "aicode",
    "count": 8,
    "children": [],
    "files": [
      {
        "title": "1",
        "path": "aicode/1.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/1.md"
      },
      {
        "title": "AGENTSmd实践指南",
        "path": "aicode/AGENTSmd实践指南.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/AGENTSmd%E5%AE%9E%E8%B7%B5%E6%8C%87%E5%8D%97.md"
      },
      {
        "title": "ai engineer时代",
        "path": "aicode/ai engineer时代.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/ai%20engineer%E6%97%B6%E4%BB%A3.md"
      },
      {
        "title": "codegraph",
        "path": "aicode/codegraph.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/codegraph.md"
      },
      {
        "title": "hooks机制",
        "path": "aicode/hooks机制.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/hooks%E6%9C%BA%E5%88%B6.md"
      },
      {
        "title": "SDD 编程实践",
        "path": "aicode/SDD 编程实践.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/SDD%20%E7%BC%96%E7%A8%8B%E5%AE%9E%E8%B7%B5.md"
      },
      {
        "title": "skills",
        "path": "aicode/skills.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/skills.md"
      },
      {
        "title": "工程技术：在智能体优先的世界中利用 Codex",
        "path": "aicode/工程技术：在智能体优先的世界中利用 Codex.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/%E5%B7%A5%E7%A8%8B%E6%8A%80%E6%9C%AF%EF%BC%9A%E5%9C%A8%E6%99%BA%E8%83%BD%E4%BD%93%E4%BC%98%E5%85%88%E7%9A%84%E4%B8%96%E7%95%8C%E4%B8%AD%E5%88%A9%E7%94%A8%20Codex.md"
      }
    ]
  },
  {
    "name": "backend",
    "path": "backend",
    "count": 20,
    "children": [
      {
        "name": "go",
        "path": "backend/go",
        "count": 18,
        "children": [],
        "files": [
          {
            "title": "Gin框架",
            "path": "backend/go/Gin框架.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/Gin%E6%A1%86%E6%9E%B6.md"
          },
          {
            "title": "Gin框架-获取参数",
            "path": "backend/go/Gin框架-获取参数.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/Gin%E6%A1%86%E6%9E%B6-%E8%8E%B7%E5%8F%96%E5%8F%82%E6%95%B0.md"
          },
          {
            "title": "GORM框架",
            "path": "backend/go/GORM框架.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/GORM%E6%A1%86%E6%9E%B6.md"
          },
          {
            "title": "go反射",
            "path": "backend/go/go反射.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%8F%8D%E5%B0%84.md"
          },
          {
            "title": "go基础",
            "path": "backend/go/go基础.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%9F%BA%E7%A1%80.md"
          },
          {
            "title": "go基础-error详解",
            "path": "backend/go/go基础-error详解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%9F%BA%E7%A1%80-error%E8%AF%A6%E8%A7%A3.md"
          },
          {
            "title": "go并发-context详解",
            "path": "backend/go/go并发-context详解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%B9%B6%E5%8F%91-context%E8%AF%A6%E8%A7%A3.md"
          },
          {
            "title": "go并发-select详解",
            "path": "backend/go/go并发-select详解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%B9%B6%E5%8F%91-select%E8%AF%A6%E8%A7%A3.md"
          },
          {
            "title": "go并发-sync、atomic",
            "path": "backend/go/go并发-sync、atomic.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%B9%B6%E5%8F%91-sync%E3%80%81atomic.md"
          },
          {
            "title": "go模块",
            "path": "backend/go/go模块.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E6%A8%A1%E5%9D%97.md"
          },
          {
            "title": "go泛型",
            "path": "backend/go/go泛型.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E6%B3%9B%E5%9E%8B.md"
          },
          {
            "title": "go相关标准库",
            "path": "backend/go/go相关标准库.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E7%9B%B8%E5%85%B3%E6%A0%87%E5%87%86%E5%BA%93.md"
          },
          {
            "title": "Go类型系统概述",
            "path": "backend/go/Go类型系统概述.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/Go%E7%B1%BB%E5%9E%8B%E7%B3%BB%E7%BB%9F%E6%A6%82%E8%BF%B0.md"
          },
          {
            "title": "go通道",
            "path": "backend/go/go通道.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E9%80%9A%E9%81%93.md"
          },
          {
            "title": "testing模块",
            "path": "backend/go/testing模块.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/testing%E6%A8%A1%E5%9D%97.md"
          },
          {
            "title": "vscode调试go程序",
            "path": "backend/go/vscode调试go程序.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/vscode%E8%B0%83%E8%AF%95go%E7%A8%8B%E5%BA%8F.md"
          },
          {
            "title": "安装",
            "path": "backend/go/安装.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/%E5%AE%89%E8%A3%85.md"
          },
          {
            "title": "资料",
            "path": "backend/go/资料.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/%E8%B5%84%E6%96%99.md"
          }
        ]
      },
      {
        "name": "mysql",
        "path": "backend/mysql",
        "count": 2,
        "children": [],
        "files": [
          {
            "title": "阶段一",
            "path": "backend/mysql/阶段一.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/mysql/%E9%98%B6%E6%AE%B5%E4%B8%80.md"
          },
          {
            "title": "阶段二",
            "path": "backend/mysql/阶段二.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/mysql/%E9%98%B6%E6%AE%B5%E4%BA%8C.md"
          }
        ]
      }
    ],
    "files": []
  },
  {
    "name": "good",
    "path": "good",
    "count": 1,
    "children": [],
    "files": [
      {
        "title": "aicodeing ref",
        "path": "good/aicodeing ref.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/good/aicodeing%20ref.md"
      }
    ]
  },
  {
    "name": "linux",
    "path": "linux",
    "count": 9,
    "children": [
      {
        "name": "docker",
        "path": "linux/docker",
        "count": 1,
        "children": [],
        "files": [
          {
            "title": "docker安装mongo",
            "path": "linux/docker/docker安装mongo.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/docker/docker%E5%AE%89%E8%A3%85mongo.md"
          }
        ]
      },
      {
        "name": "git",
        "path": "linux/git",
        "count": 3,
        "children": [],
        "files": [
          {
            "title": "git常用操作指令",
            "path": "linux/git/git常用操作指令.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/git/git%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C%E6%8C%87%E4%BB%A4.md"
          },
          {
            "title": "git项目大改",
            "path": "linux/git/git项目大改.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/git/git%E9%A1%B9%E7%9B%AE%E5%A4%A7%E6%94%B9.md"
          },
          {
            "title": "本地仓库连接到github",
            "path": "linux/git/本地仓库连接到github.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/git/%7F%E6%9C%AC%E5%9C%B0%E4%BB%93%E5%BA%93%E8%BF%9E%E6%8E%A5%E5%88%B0github.md"
          }
        ]
      }
    ],
    "files": [
      {
        "title": "claude+chatglm4.6",
        "path": "linux/claude+chatglm4.6.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/claude%2Bchatglm4.6.md"
      },
      {
        "title": "github-action博客页面",
        "path": "linux/github-action博客页面.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/github-action%E5%8D%9A%E5%AE%A2%E9%A1%B5%E9%9D%A2.md"
      },
      {
        "title": "linux、windows协同工作注意",
        "path": "linux/linux、windows协同工作注意.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/linux%E3%80%81windows%E5%8D%8F%E5%90%8C%E5%B7%A5%E4%BD%9C%E6%B3%A8%E6%84%8F.md"
      },
      {
        "title": "macos配置zsh",
        "path": "linux/macos配置zsh.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/macos%E9%85%8D%E7%BD%AEzsh.md"
      },
      {
        "title": "vscode-ssh免密登录",
        "path": "linux/vscode-ssh免密登录.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/vscode-ssh%E5%85%8D%E5%AF%86%E7%99%BB%E5%BD%95.md"
      }
    ]
  },
  {
    "name": "llm",
    "path": "llm",
    "count": 31,
    "children": [
      {
        "name": "agent",
        "path": "llm/agent",
        "count": 14,
        "children": [],
        "files": [
          {
            "title": "agent时代的CLI",
            "path": "llm/agent/agent时代的CLI.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/agent%E6%97%B6%E4%BB%A3%E7%9A%84CLI.md"
          },
          {
            "title": "claude code源码",
            "path": "llm/agent/claude code源码.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/claude%20code%E6%BA%90%E7%A0%81.md"
          },
          {
            "title": "datawhale：01 agent应用开发与落地全景",
            "path": "llm/agent/datawhale：01 agent应用开发与落地全景.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/datawhale%EF%BC%9A01%20agent%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E4%B8%8E%E8%90%BD%E5%9C%B0%E5%85%A8%E6%99%AF.md"
          },
          {
            "title": "datawhale：02 agent原理与最简实践",
            "path": "llm/agent/datawhale：02 agent原理与最简实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/datawhale%EF%BC%9A02%20agent%E5%8E%9F%E7%90%86%E4%B8%8E%E6%9C%80%E7%AE%80%E5%AE%9E%E8%B7%B5.md"
          },
          {
            "title": "datawhale：03 多智能体开发范式与最佳实践",
            "path": "llm/agent/datawhale：03 多智能体开发范式与最佳实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/datawhale%EF%BC%9A03%20%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E5%BC%80%E5%8F%91%E8%8C%83%E5%BC%8F%E4%B8%8E%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md"
          },
          {
            "title": "harness不是目的，知识才是护城河",
            "path": "llm/agent/harness不是目的，知识才是护城河.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/harness%E4%B8%8D%E6%98%AF%E7%9B%AE%E7%9A%84%EF%BC%8C%E7%9F%A5%E8%AF%86%E6%89%8D%E6%98%AF%E6%8A%A4%E5%9F%8E%E6%B2%B3.md"
          },
          {
            "title": "Harness理解",
            "path": "llm/agent/Harness理解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/Harness%E7%90%86%E8%A7%A3.md"
          },
          {
            "title": "llm agent应用实践",
            "path": "llm/agent/llm agent应用实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/llm%20agent%E5%BA%94%E7%94%A8%E5%AE%9E%E8%B7%B5.md"
          },
          {
            "title": "llm agent提示词应用实践",
            "path": "llm/agent/llm agent提示词应用实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/llm%20agent%E6%8F%90%E7%A4%BA%E8%AF%8D%E5%BA%94%E7%94%A8%E5%AE%9E%E8%B7%B5.md"
          },
          {
            "title": "memori agent的记忆引擎",
            "path": "llm/agent/memori agent的记忆引擎.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/memori%20agent%E7%9A%84%E8%AE%B0%E5%BF%86%E5%BC%95%E6%93%8E.md"
          },
          {
            "title": "opencode skills实现原理",
            "path": "llm/agent/opencode skills实现原理.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/opencode%20skills%E5%AE%9E%E7%8E%B0%E5%8E%9F%E7%90%86.md"
          },
          {
            "title": "意图识别、槽位填充，参数提取节点",
            "path": "llm/agent/意图识别、槽位填充，参数提取节点.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E6%84%8F%E5%9B%BE%E8%AF%86%E5%88%AB%E3%80%81%E6%A7%BD%E4%BD%8D%E5%A1%AB%E5%85%85%EF%BC%8C%E5%8F%82%E6%95%B0%E6%8F%90%E5%8F%96%E8%8A%82%E7%82%B9.md"
          },
          {
            "title": "耿直哥_智能体狂欢之后，谁还值钱？｜ 5月AI行业洞察",
            "path": "llm/agent/耿直哥_智能体狂欢之后，谁还值钱？｜ 5月AI行业洞察.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E8%80%BF%E7%9B%B4%E5%93%A5_%E6%99%BA%E8%83%BD%E4%BD%93%E7%8B%82%E6%AC%A2%E4%B9%8B%E5%90%8E%EF%BC%8C%E8%B0%81%E8%BF%98%E5%80%BC%E9%92%B1%EF%BC%9F%EF%BD%9C%205%E6%9C%88AI%E8%A1%8C%E4%B8%9A%E6%B4%9E%E5%AF%9F.md"
          },
          {
            "title": "资料",
            "path": "llm/agent/资料.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E8%B5%84%E6%96%99.md"
          }
        ]
      },
      {
        "name": "rag",
        "path": "llm/rag",
        "count": 1,
        "children": [],
        "files": [
          {
            "title": "embedding模型选取",
            "path": "llm/rag/embedding模型选取.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/rag/embedding%E6%A8%A1%E5%9E%8B%E9%80%89%E5%8F%96.md"
          }
        ]
      },
      {
        "name": "sft",
        "path": "llm/sft",
        "count": 5,
        "children": [],
        "files": [
          {
            "title": "lora、qlora微调",
            "path": "llm/sft/lora、qlora微调.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/lora%E3%80%81qlora%E5%BE%AE%E8%B0%83.md"
          },
          {
            "title": "ms-swift微调qwen3-0.6b模型",
            "path": "llm/sft/ms-swift微调qwen3-0.6b模型.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/ms-swift%E5%BE%AE%E8%B0%83qwen3-0.6b%E6%A8%A1%E5%9E%8B.md"
          },
          {
            "title": "qwen3-4b lora微调",
            "path": "llm/sft/qwen3-4b lora微调.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/qwen3-4b%20lora%E5%BE%AE%E8%B0%83.md"
          },
          {
            "title": "qwen3-8b lora微调",
            "path": "llm/sft/qwen3-8b lora微调.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/qwen3-8b%20lora%E5%BE%AE%E8%B0%83.md"
          },
          {
            "title": "改提示词还是微调",
            "path": "llm/sft/改提示词还是微调.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/%E6%94%B9%E6%8F%90%E7%A4%BA%E8%AF%8D%E8%BF%98%E6%98%AF%E5%BE%AE%E8%B0%83.md"
          }
        ]
      },
      {
        "name": "theory",
        "path": "llm/theory",
        "count": 11,
        "children": [],
        "files": [
          {
            "title": "agentic rl",
            "path": "llm/theory/agentic rl.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/agentic%20rl.md"
          },
          {
            "title": "deepseek r1技术报告",
            "path": "llm/theory/deepseek r1技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/deepseek%20r1%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md"
          },
          {
            "title": "deeqseek v3.2 技术报告",
            "path": "llm/theory/deeqseek v3.2 技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/deeqseek%20v3.2%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md"
          },
          {
            "title": "MoE概念",
            "path": "llm/theory/MoE概念.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/MoE%E6%A6%82%E5%BF%B5.md"
          },
          {
            "title": "qwen2.5 技术报告",
            "path": "llm/theory/qwen2.5 技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/qwen2.5%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md"
          },
          {
            "title": "qwen3 技术报告",
            "path": "llm/theory/qwen3 技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/qwen3%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md"
          },
          {
            "title": "transformer",
            "path": "llm/theory/transformer.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/transformer.md"
          },
          {
            "title": "vllm等推理框架的优化",
            "path": "llm/theory/vllm等推理框架的优化.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/vllm%E7%AD%89%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6%E7%9A%84%E4%BC%98%E5%8C%96.md"
          },
          {
            "title": "上下文工程",
            "path": "llm/theory/上下文工程.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B.md"
          },
          {
            "title": "多模态大模型",
            "path": "llm/theory/多模态大模型.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/%E5%A4%9A%E6%A8%A1%E6%80%81%E5%A4%A7%E6%A8%A1%E5%9E%8B.md"
          },
          {
            "title": "大模型基础",
            "path": "llm/theory/大模型基础.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80.md"
          }
        ]
      }
    ],
    "files": []
  },
  {
    "name": "ml",
    "path": "ml",
    "count": 3,
    "children": [],
    "files": [
      {
        "title": "pytorch模型训练",
        "path": "ml/pytorch模型训练.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/ml/pytorch%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83.md"
      },
      {
        "title": "XGBoost",
        "path": "ml/XGBoost.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/ml/XGBoost.md"
      },
      {
        "title": "机器学习的评价指标",
        "path": "ml/机器学习的评价指标.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/ml/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87.md"
      }
    ]
  },
  {
    "name": "others",
    "path": "others",
    "count": 3,
    "children": [],
    "files": [
      {
        "title": "golemon-blogs页面",
        "path": "others/golemon-blogs页面.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/others/golemon-blogs%E9%A1%B5%E9%9D%A2.md"
      },
      {
        "title": "how read paper",
        "path": "others/how read paper.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/others/how%20read%20paper.md"
      },
      {
        "title": "llm应用相关的简易内容",
        "path": "others/llm应用相关的简易内容.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/others/llm%E5%BA%94%E7%94%A8%E7%9B%B8%E5%85%B3%E7%9A%84%E7%AE%80%E6%98%93%E5%86%85%E5%AE%B9.md"
      }
    ]
  },
  {
    "name": "skills",
    "path": "skills",
    "count": 5,
    "children": [
      {
        "name": "course",
        "path": "skills/course",
        "count": 5,
        "children": [
          {
            "name": "docs",
            "path": "skills/course/docs",
            "count": 1,
            "children": [],
            "files": [
              {
                "title": "usage",
                "path": "skills/course/docs/usage.md",
                "url": "https://github.com/golitter/glm-blogs/blob/master/skills/course/docs/usage.md"
              }
            ]
          },
          {
            "name": "sub",
            "path": "skills/course/sub",
            "count": 3,
            "children": [],
            "files": [
              {
                "title": "content",
                "path": "skills/course/sub/content.md",
                "url": "https://github.com/golitter/glm-blogs/blob/master/skills/course/sub/content.md"
              },
              {
                "title": "plan",
                "path": "skills/course/sub/plan.md",
                "url": "https://github.com/golitter/glm-blogs/blob/master/skills/course/sub/plan.md"
              },
              {
                "title": "summary",
                "path": "skills/course/sub/summary.md",
                "url": "https://github.com/golitter/glm-blogs/blob/master/skills/course/sub/summary.md"
              }
            ]
          }
        ],
        "files": [
          {
            "title": "SKILL",
            "path": "skills/course/SKILL.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/skills/course/SKILL.md"
          }
        ]
      }
    ],
    "files": []
  },
  {
    "name": "wm",
    "path": "wm",
    "count": 2,
    "children": [],
    "files": [
      {
        "title": "LLM到World Model",
        "path": "wm/LLM到World Model.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/wm/LLM%E5%88%B0World%20Model.md"
      },
      {
        "title": "世界模型入门",
        "path": "wm/世界模型入门.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/wm/%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8.md"
      }
    ]
  }
] satisfies BlogTreeNode[];
export const recentFiles = [
  {
    "title": "golemon-blogs页面",
    "path": "others/golemon-blogs页面.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/others/golemon-blogs%E9%A1%B5%E9%9D%A2.md",
    "date": "2026-07-03 15:13"
  },
  {
    "title": "世界模型入门",
    "path": "wm/世界模型入门.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/wm/%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8.md",
    "date": "2026-07-02 23:59"
  },
  {
    "title": "LLM到World Model",
    "path": "wm/LLM到World Model.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/wm/LLM%E5%88%B0World%20Model.md",
    "date": "2026-07-02 23:22"
  },
  {
    "title": "go相关标准库",
    "path": "backend/go/go相关标准库.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E7%9B%B8%E5%85%B3%E6%A0%87%E5%87%86%E5%BA%93.md",
    "date": "2026-06-29 15:52"
  },
  {
    "title": "go基础-error详解",
    "path": "backend/go/go基础-error详解.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%9F%BA%E7%A1%80-error%E8%AF%A6%E8%A7%A3.md",
    "date": "2026-06-27 14:49"
  }
] satisfies RecentFile[];
export const markdownCount = 82;
export const updateTime = "2026-07-03 16:36:33";
