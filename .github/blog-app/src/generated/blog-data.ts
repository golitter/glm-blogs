export type BlogFile = {
  title: string;
  path: string;
  url: string;
  updatedAt?: number;
  change?: "added" | "modified";
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
    "count": 11,
    "children": [],
    "files": [
      {
        "title": "1",
        "path": "aicode/1.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/1.md",
        "updatedAt": 1774165728000,
        "change": "added"
      },
      {
        "title": "AGENTSmd实践指南",
        "path": "aicode/AGENTSmd实践指南.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/AGENTSmd%E5%AE%9E%E8%B7%B5%E6%8C%87%E5%8D%97.md",
        "updatedAt": 1781415562000,
        "change": "modified"
      },
      {
        "title": "ai engineer时代",
        "path": "aicode/ai engineer时代.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/ai%20engineer%E6%97%B6%E4%BB%A3.md",
        "updatedAt": 1777964947000,
        "change": "added"
      },
      {
        "title": "AI Native时代，如何更好的用ai工具",
        "path": "aicode/AI Native时代，如何更好的用ai工具.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/AI%20Native%E6%97%B6%E4%BB%A3%EF%BC%8C%E5%A6%82%E4%BD%95%E6%9B%B4%E5%A5%BD%E7%9A%84%E7%94%A8ai%E5%B7%A5%E5%85%B7.md",
        "updatedAt": 1783095938000,
        "change": "modified"
      },
      {
        "title": "aicoding agent会话成本管控",
        "path": "aicode/aicoding agent会话成本管控.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/aicoding%20agent%E4%BC%9A%E8%AF%9D%E6%88%90%E6%9C%AC%E7%AE%A1%E6%8E%A7.md",
        "updatedAt": 1786968393000,
        "change": "added"
      },
      {
        "title": "codegraph",
        "path": "aicode/codegraph.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/codegraph.md",
        "updatedAt": 1781419348000,
        "change": "added"
      },
      {
        "title": "hooks机制",
        "path": "aicode/hooks机制.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/hooks%E6%9C%BA%E5%88%B6.md",
        "updatedAt": 1781420968000,
        "change": "added"
      },
      {
        "title": "SDD 编程实践",
        "path": "aicode/SDD 编程实践.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/SDD%20%E7%BC%96%E7%A8%8B%E5%AE%9E%E8%B7%B5.md",
        "updatedAt": 1778222253000,
        "change": "added"
      },
      {
        "title": "skills",
        "path": "aicode/skills.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/skills.md",
        "updatedAt": 1774165728000,
        "change": "added"
      },
      {
        "title": "字节2026-ai全栈挑战赛ai协助开发记录",
        "path": "aicode/字节2026-ai全栈挑战赛ai协助开发记录.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/%E5%AD%97%E8%8A%822026-ai%E5%85%A8%E6%A0%88%E6%8C%91%E6%88%98%E8%B5%9Bai%E5%8D%8F%E5%8A%A9%E5%BC%80%E5%8F%91%E8%AE%B0%E5%BD%95.md",
        "updatedAt": 1783095662000,
        "change": "added"
      },
      {
        "title": "工程技术：在智能体优先的世界中利用 Codex",
        "path": "aicode/工程技术：在智能体优先的世界中利用 Codex.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/%E5%B7%A5%E7%A8%8B%E6%8A%80%E6%9C%AF%EF%BC%9A%E5%9C%A8%E6%99%BA%E8%83%BD%E4%BD%93%E4%BC%98%E5%85%88%E7%9A%84%E4%B8%96%E7%95%8C%E4%B8%AD%E5%88%A9%E7%94%A8%20Codex.md",
        "updatedAt": 1778222253000,
        "change": "added"
      }
    ]
  },
  {
    "name": "backend",
    "path": "backend",
    "count": 31,
    "children": [
      {
        "name": "go",
        "path": "backend/go",
        "count": 21,
        "children": [],
        "files": [
          {
            "title": "Gin框架",
            "path": "backend/go/Gin框架.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/Gin%E6%A1%86%E6%9E%B6.md",
            "updatedAt": 1777973947000,
            "change": "modified"
          },
          {
            "title": "Gin框架-获取参数",
            "path": "backend/go/Gin框架-获取参数.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/Gin%E6%A1%86%E6%9E%B6-%E8%8E%B7%E5%8F%96%E5%8F%82%E6%95%B0.md",
            "updatedAt": 1777973947000,
            "change": "added"
          },
          {
            "title": "GMP",
            "path": "backend/go/GMP.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/GMP.md",
            "updatedAt": 1783925673000,
            "change": "modified"
          },
          {
            "title": "GORM-CRUD查询分页删除批量操作",
            "path": "backend/go/GORM-CRUD查询分页删除批量操作.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/GORM-CRUD%E6%9F%A5%E8%AF%A2%E5%88%86%E9%A1%B5%E5%88%A0%E9%99%A4%E6%89%B9%E9%87%8F%E6%93%8D%E4%BD%9C.md",
            "updatedAt": 1783496499000,
            "change": "added"
          },
          {
            "title": "GORM框架",
            "path": "backend/go/GORM框架.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/GORM%E6%A1%86%E6%9E%B6.md",
            "updatedAt": 1783439395000,
            "change": "modified"
          },
          {
            "title": "go反射",
            "path": "backend/go/go反射.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%8F%8D%E5%B0%84.md",
            "updatedAt": 1782381069000,
            "change": "modified"
          },
          {
            "title": "go基础",
            "path": "backend/go/go基础.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%9F%BA%E7%A1%80.md",
            "updatedAt": 1782138576000,
            "change": "modified"
          },
          {
            "title": "go基础-error详解",
            "path": "backend/go/go基础-error详解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%9F%BA%E7%A1%80-error%E8%AF%A6%E8%A7%A3.md",
            "updatedAt": 1782542968000,
            "change": "added"
          },
          {
            "title": "go并发-context详解",
            "path": "backend/go/go并发-context详解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%B9%B6%E5%8F%91-context%E8%AF%A6%E8%A7%A3.md",
            "updatedAt": 1783773687000,
            "change": "modified"
          },
          {
            "title": "go并发-select详解",
            "path": "backend/go/go并发-select详解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%B9%B6%E5%8F%91-select%E8%AF%A6%E8%A7%A3.md",
            "updatedAt": 1782487861000,
            "change": "added"
          },
          {
            "title": "go并发-sync、atomic",
            "path": "backend/go/go并发-sync、atomic.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E5%B9%B6%E5%8F%91-sync%E3%80%81atomic.md",
            "updatedAt": 1782469805000,
            "change": "added"
          },
          {
            "title": "go模块",
            "path": "backend/go/go模块.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E6%A8%A1%E5%9D%97.md",
            "updatedAt": 1777706832000,
            "change": "added"
          },
          {
            "title": "go泛型",
            "path": "backend/go/go泛型.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E6%B3%9B%E5%9E%8B.md",
            "updatedAt": 1777468811000,
            "change": "added"
          },
          {
            "title": "go相关标准库",
            "path": "backend/go/go相关标准库.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E7%9B%B8%E5%85%B3%E6%A0%87%E5%87%86%E5%BA%93.md",
            "updatedAt": 1783316792000,
            "change": "modified"
          },
          {
            "title": "Go类型系统概述",
            "path": "backend/go/Go类型系统概述.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/Go%E7%B1%BB%E5%9E%8B%E7%B3%BB%E7%BB%9F%E6%A6%82%E8%BF%B0.md",
            "updatedAt": 1781882724000,
            "change": "added"
          },
          {
            "title": "go通道",
            "path": "backend/go/go通道.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/go%E9%80%9A%E9%81%93.md",
            "updatedAt": 1782229892000,
            "change": "added"
          },
          {
            "title": "testing模块",
            "path": "backend/go/testing模块.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/testing%E6%A8%A1%E5%9D%97.md",
            "updatedAt": 1784963360000,
            "change": "modified"
          },
          {
            "title": "Viper配置管理",
            "path": "backend/go/Viper配置管理.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/Viper%E9%85%8D%E7%BD%AE%E7%AE%A1%E7%90%86.md",
            "updatedAt": 1783778378000,
            "change": "added"
          },
          {
            "title": "vscode调试go程序",
            "path": "backend/go/vscode调试go程序.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/vscode%E8%B0%83%E8%AF%95go%E7%A8%8B%E5%BA%8F.md",
            "updatedAt": 1777885209000,
            "change": "added"
          },
          {
            "title": "安装",
            "path": "backend/go/安装.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/%E5%AE%89%E8%A3%85.md",
            "updatedAt": 1777706832000,
            "change": "modified"
          },
          {
            "title": "资料",
            "path": "backend/go/资料.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/go/%E8%B5%84%E6%96%99.md",
            "updatedAt": 1783841012000,
            "change": "modified"
          }
        ]
      },
      {
        "name": "mysql",
        "path": "backend/mysql",
        "count": 5,
        "children": [],
        "files": [
          {
            "title": "InnoDB架构与日志",
            "path": "backend/mysql/InnoDB架构与日志.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/mysql/InnoDB%E6%9E%B6%E6%9E%84%E4%B8%8E%E6%97%A5%E5%BF%97.md",
            "updatedAt": 1783317437000,
            "change": "added"
          },
          {
            "title": "事务锁与MVCC",
            "path": "backend/mysql/事务锁与MVCC.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/mysql/%E4%BA%8B%E5%8A%A1%E9%94%81%E4%B8%8EMVCC.md",
            "updatedAt": 1783317437000,
            "change": "added"
          },
          {
            "title": "查询语法与索引基础",
            "path": "backend/mysql/查询语法与索引基础.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/mysql/%E6%9F%A5%E8%AF%A2%E8%AF%AD%E6%B3%95%E4%B8%8E%E7%B4%A2%E5%BC%95%E5%9F%BA%E7%A1%80.md",
            "updatedAt": 1783317437000,
            "change": "added"
          },
          {
            "title": "索引原理与执行计划",
            "path": "backend/mysql/索引原理与执行计划.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/mysql/%E7%B4%A2%E5%BC%95%E5%8E%9F%E7%90%86%E4%B8%8E%E6%89%A7%E8%A1%8C%E8%AE%A1%E5%88%92.md",
            "updatedAt": 1783317437000,
            "change": "added"
          },
          {
            "title": "表设计与运维优化",
            "path": "backend/mysql/表设计与运维优化.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/mysql/%E8%A1%A8%E8%AE%BE%E8%AE%A1%E4%B8%8E%E8%BF%90%E7%BB%B4%E4%BC%98%E5%8C%96.md",
            "updatedAt": 1783317437000,
            "change": "added"
          }
        ]
      },
      {
        "name": "redis",
        "path": "backend/redis",
        "count": 3,
        "children": [],
        "files": [
          {
            "title": "持久化与内存管理",
            "path": "backend/redis/持久化与内存管理.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/redis/%E6%8C%81%E4%B9%85%E5%8C%96%E4%B8%8E%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86.md",
            "updatedAt": 1784613901000,
            "change": "modified"
          },
          {
            "title": "数据类型",
            "path": "backend/redis/数据类型.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/redis/%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B.md",
            "updatedAt": 1784549971000,
            "change": "modified"
          },
          {
            "title": "缓存设计",
            "path": "backend/redis/缓存设计.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/backend/redis/%E7%BC%93%E5%AD%98%E8%AE%BE%E8%AE%A1.md",
            "updatedAt": 1784543208000,
            "change": "added"
          }
        ]
      }
    ],
    "files": [
      {
        "title": "RESTful API",
        "path": "backend/RESTful API.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/backend/RESTful%20API.md",
        "updatedAt": 1783930600000,
        "change": "modified"
      },
      {
        "title": "后端测试分层与压测",
        "path": "backend/后端测试分层与压测.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/backend/%E5%90%8E%E7%AB%AF%E6%B5%8B%E8%AF%95%E5%88%86%E5%B1%82%E4%B8%8E%E5%8E%8B%E6%B5%8B.md",
        "updatedAt": 1785228642000,
        "change": "added"
      }
    ]
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
        "url": "https://github.com/golitter/glm-blogs/blob/master/good/aicodeing%20ref.md",
        "updatedAt": 1774161377000,
        "change": "added"
      }
    ]
  },
  {
    "name": "linux",
    "path": "linux",
    "count": 12,
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
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/docker/docker%E5%AE%89%E8%A3%85mongo.md",
            "updatedAt": 1763439137000,
            "change": "added"
          }
        ]
      },
      {
        "name": "git",
        "path": "linux/git",
        "count": 4,
        "children": [],
        "files": [
          {
            "title": "git冲突处理指南",
            "path": "linux/git/git冲突处理指南.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/git/git%E5%86%B2%E7%AA%81%E5%A4%84%E7%90%86%E6%8C%87%E5%8D%97.md",
            "updatedAt": 1785664310000,
            "change": "added"
          },
          {
            "title": "git常用操作指令",
            "path": "linux/git/git常用操作指令.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/git/git%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C%E6%8C%87%E4%BB%A4.md",
            "updatedAt": 1762064056000,
            "change": "added"
          },
          {
            "title": "git项目大改",
            "path": "linux/git/git项目大改.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/linux/git/git%E9%A1%B9%E7%9B%AE%E5%A4%A7%E6%94%B9.md",
            "updatedAt": 1765101743000,
            "change": "modified"
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
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/claude%2Bchatglm4.6.md",
        "updatedAt": 1764034626000,
        "change": "modified"
      },
      {
        "title": "Codex连接本地WSL2-SSH教程",
        "path": "linux/Codex连接本地WSL2-SSH教程.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/Codex%E8%BF%9E%E6%8E%A5%E6%9C%AC%E5%9C%B0WSL2-SSH%E6%95%99%E7%A8%8B.md",
        "updatedAt": 1784786869000,
        "change": "modified"
      },
      {
        "title": "github-action博客页面",
        "path": "linux/github-action博客页面.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/github-action%E5%8D%9A%E5%AE%A2%E9%A1%B5%E9%9D%A2.md",
        "updatedAt": 1763469168000,
        "change": "added"
      },
      {
        "title": "linux、windows协同工作注意",
        "path": "linux/linux、windows协同工作注意.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/linux%E3%80%81windows%E5%8D%8F%E5%90%8C%E5%B7%A5%E4%BD%9C%E6%B3%A8%E6%84%8F.md",
        "updatedAt": 1763468329000,
        "change": "added"
      },
      {
        "title": "macos配置zsh",
        "path": "linux/macos配置zsh.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/macos%E9%85%8D%E7%BD%AEzsh.md",
        "updatedAt": 1777528541000,
        "change": "added"
      },
      {
        "title": "tmux常用命令",
        "path": "linux/tmux常用命令.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/tmux%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4.md",
        "updatedAt": 1784799111000,
        "change": "added"
      },
      {
        "title": "vscode-ssh免密登录",
        "path": "linux/vscode-ssh免密登录.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/linux/vscode-ssh%E5%85%8D%E5%AF%86%E7%99%BB%E5%BD%95.md",
        "updatedAt": 1762062153000,
        "change": "added"
      }
    ]
  },
  {
    "name": "llm",
    "path": "llm",
    "count": 41,
    "children": [
      {
        "name": "agent",
        "path": "llm/agent",
        "count": 23,
        "children": [],
        "files": [
          {
            "title": "agent从记忆到自我进化",
            "path": "llm/agent/agent从记忆到自我进化.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/agent%E4%BB%8E%E8%AE%B0%E5%BF%86%E5%88%B0%E8%87%AA%E6%88%91%E8%BF%9B%E5%8C%96.md",
            "updatedAt": 1784445078000,
            "change": "added"
          },
          {
            "title": "agent时代的CLI",
            "path": "llm/agent/agent时代的CLI.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/agent%E6%97%B6%E4%BB%A3%E7%9A%84CLI.md",
            "updatedAt": 1774953267000,
            "change": "added"
          },
          {
            "title": "agent的意图识别设计",
            "path": "llm/agent/agent的意图识别设计.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/agent%E7%9A%84%E6%84%8F%E5%9B%BE%E8%AF%86%E5%88%AB%E8%AE%BE%E8%AE%A1.md",
            "updatedAt": 1785496808000,
            "change": "added"
          },
          {
            "title": "agent评测",
            "path": "llm/agent/agent评测.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/agent%E8%AF%84%E6%B5%8B.md",
            "updatedAt": 1783092006000,
            "change": "added"
          },
          {
            "title": "Agent高危工具与长时间工具调用设计",
            "path": "llm/agent/Agent高危工具与长时间工具调用设计.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/Agent%E9%AB%98%E5%8D%B1%E5%B7%A5%E5%85%B7%E4%B8%8E%E9%95%BF%E6%97%B6%E9%97%B4%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E8%AE%BE%E8%AE%A1.md",
            "updatedAt": 1785239586000,
            "change": "modified"
          },
          {
            "title": "aicoding memory实现",
            "path": "llm/agent/aicoding memory实现.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/aicoding%20memory%E5%AE%9E%E7%8E%B0.md",
            "updatedAt": 1786694490000,
            "change": "added"
          },
          {
            "title": "claude code源码",
            "path": "llm/agent/claude code源码.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/claude%20code%E6%BA%90%E7%A0%81.md",
            "updatedAt": 1775028573000,
            "change": "added"
          },
          {
            "title": "datawhale：01 agent应用开发与落地全景",
            "path": "llm/agent/datawhale：01 agent应用开发与落地全景.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/datawhale%EF%BC%9A01%20agent%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E4%B8%8E%E8%90%BD%E5%9C%B0%E5%85%A8%E6%99%AF.md",
            "updatedAt": 1768215821000,
            "change": "added"
          },
          {
            "title": "datawhale：02 agent原理与最简实践",
            "path": "llm/agent/datawhale：02 agent原理与最简实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/datawhale%EF%BC%9A02%20agent%E5%8E%9F%E7%90%86%E4%B8%8E%E6%9C%80%E7%AE%80%E5%AE%9E%E8%B7%B5.md",
            "updatedAt": 1768634653000,
            "change": "added"
          },
          {
            "title": "datawhale：03 多智能体开发范式与最佳实践",
            "path": "llm/agent/datawhale：03 多智能体开发范式与最佳实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/datawhale%EF%BC%9A03%20%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E5%BC%80%E5%8F%91%E8%8C%83%E5%BC%8F%E4%B8%8E%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md",
            "updatedAt": 1768912522000,
            "change": "added"
          },
          {
            "title": "deepseek harness",
            "path": "llm/agent/deepseek harness.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/deepseek%20harness.md",
            "updatedAt": 1786879511000,
            "change": "added"
          },
          {
            "title": "GPT6-Astra分析",
            "path": "llm/agent/GPT6-Astra分析.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/GPT6-Astra%E5%88%86%E6%9E%90.md",
            "updatedAt": 1788606121000,
            "change": "added"
          },
          {
            "title": "harness不是目的，知识才是护城河",
            "path": "llm/agent/harness不是目的，知识才是护城河.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/harness%E4%B8%8D%E6%98%AF%E7%9B%AE%E7%9A%84%EF%BC%8C%E7%9F%A5%E8%AF%86%E6%89%8D%E6%98%AF%E6%8A%A4%E5%9F%8E%E6%B2%B3.md",
            "updatedAt": 1778649890000,
            "change": "added"
          },
          {
            "title": "Harness理解",
            "path": "llm/agent/Harness理解.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/Harness%E7%90%86%E8%A7%A3.md",
            "updatedAt": 1784365231000,
            "change": "modified"
          },
          {
            "title": "llm agent应用实践",
            "path": "llm/agent/llm agent应用实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/llm%20agent%E5%BA%94%E7%94%A8%E5%AE%9E%E8%B7%B5.md",
            "updatedAt": 1764074378000,
            "change": "added"
          },
          {
            "title": "llm agent提示词应用实践",
            "path": "llm/agent/llm agent提示词应用实践.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/llm%20agent%E6%8F%90%E7%A4%BA%E8%AF%8D%E5%BA%94%E7%94%A8%E5%AE%9E%E8%B7%B5.md",
            "updatedAt": 1765101606000,
            "change": "added"
          },
          {
            "title": "loop engineer",
            "path": "llm/agent/loop engineer.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/loop%20engineer.md",
            "updatedAt": 1783240855000,
            "change": "added"
          },
          {
            "title": "memori agent的记忆引擎",
            "path": "llm/agent/memori agent的记忆引擎.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/memori%20agent%E7%9A%84%E8%AE%B0%E5%BF%86%E5%BC%95%E6%93%8E.md",
            "updatedAt": 1764594785000,
            "change": "added"
          },
          {
            "title": "opencode skills实现原理",
            "path": "llm/agent/opencode skills实现原理.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/opencode%20skills%E5%AE%9E%E7%8E%B0%E5%8E%9F%E7%90%86.md",
            "updatedAt": 1777339537000,
            "change": "modified"
          },
          {
            "title": "开源aicoding功能原理",
            "path": "llm/agent/开源aicoding功能原理.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E5%BC%80%E6%BA%90aicoding%E5%8A%9F%E8%83%BD%E5%8E%9F%E7%90%86.md",
            "updatedAt": 1786694490000,
            "change": "added"
          },
          {
            "title": "意图识别、槽位填充，参数提取节点",
            "path": "llm/agent/意图识别、槽位填充，参数提取节点.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E6%84%8F%E5%9B%BE%E8%AF%86%E5%88%AB%E3%80%81%E6%A7%BD%E4%BD%8D%E5%A1%AB%E5%85%85%EF%BC%8C%E5%8F%82%E6%95%B0%E6%8F%90%E5%8F%96%E8%8A%82%E7%82%B9.md",
            "updatedAt": 1774190080000,
            "change": "added"
          },
          {
            "title": "耿直哥_智能体狂欢之后，谁还值钱？｜ 5月AI行业洞察",
            "path": "llm/agent/耿直哥_智能体狂欢之后，谁还值钱？｜ 5月AI行业洞察.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E8%80%BF%E7%9B%B4%E5%93%A5_%E6%99%BA%E8%83%BD%E4%BD%93%E7%8B%82%E6%AC%A2%E4%B9%8B%E5%90%8E%EF%BC%8C%E8%B0%81%E8%BF%98%E5%80%BC%E9%92%B1%EF%BC%9F%EF%BD%9C%205%E6%9C%88AI%E8%A1%8C%E4%B8%9A%E6%B4%9E%E5%AF%9F.md",
            "updatedAt": 1778838726000,
            "change": "added"
          },
          {
            "title": "资料",
            "path": "llm/agent/资料.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E8%B5%84%E6%96%99.md",
            "updatedAt": 1788522966000,
            "change": "modified"
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
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/rag/embedding%E6%A8%A1%E5%9E%8B%E9%80%89%E5%8F%96.md",
            "updatedAt": 1775037973000,
            "change": "added"
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
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/lora%E3%80%81qlora%E5%BE%AE%E8%B0%83.md",
            "updatedAt": 1762676012000,
            "change": "added"
          },
          {
            "title": "ms-swift微调qwen3-0.6b模型",
            "path": "llm/sft/ms-swift微调qwen3-0.6b模型.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/ms-swift%E5%BE%AE%E8%B0%83qwen3-0.6b%E6%A8%A1%E5%9E%8B.md",
            "updatedAt": 1762062153000,
            "change": "added"
          },
          {
            "title": "qwen3-4b lora微调",
            "path": "llm/sft/qwen3-4b lora微调.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/qwen3-4b%20lora%E5%BE%AE%E8%B0%83.md",
            "updatedAt": 1762135263000,
            "change": "added"
          },
          {
            "title": "qwen3-8b lora微调",
            "path": "llm/sft/qwen3-8b lora微调.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/qwen3-8b%20lora%E5%BE%AE%E8%B0%83.md",
            "updatedAt": 1763455328000,
            "change": "modified"
          },
          {
            "title": "改提示词还是微调",
            "path": "llm/sft/改提示词还是微调.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/sft/%E6%94%B9%E6%8F%90%E7%A4%BA%E8%AF%8D%E8%BF%98%E6%98%AF%E5%BE%AE%E8%B0%83.md",
            "updatedAt": 1764661437000,
            "change": "added"
          }
        ]
      },
      {
        "name": "theory",
        "path": "llm/theory",
        "count": 12,
        "children": [],
        "files": [
          {
            "title": "agentic rl",
            "path": "llm/theory/agentic rl.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/agentic%20rl.md",
            "updatedAt": 1763448040000,
            "change": "added"
          },
          {
            "title": "deepseek r1技术报告",
            "path": "llm/theory/deepseek r1技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/deepseek%20r1%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md",
            "updatedAt": 1764074378000,
            "change": "added"
          },
          {
            "title": "deeqseek v3.2 技术报告",
            "path": "llm/theory/deeqseek v3.2 技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/deeqseek%20v3.2%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md",
            "updatedAt": 1764657335000,
            "change": "added"
          },
          {
            "title": "J-Space",
            "path": "llm/theory/J-Space.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/J-Space.md",
            "updatedAt": 1786966513000,
            "change": "added"
          },
          {
            "title": "MoE概念",
            "path": "llm/theory/MoE概念.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/MoE%E6%A6%82%E5%BF%B5.md",
            "updatedAt": 1764074378000,
            "change": "added"
          },
          {
            "title": "qwen2.5 技术报告",
            "path": "llm/theory/qwen2.5 技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/qwen2.5%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md",
            "updatedAt": 1764488935000,
            "change": "added"
          },
          {
            "title": "qwen3 技术报告",
            "path": "llm/theory/qwen3 技术报告.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/qwen3%20%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A.md",
            "updatedAt": 1764403035000,
            "change": "added"
          },
          {
            "title": "transformer",
            "path": "llm/theory/transformer.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/transformer.md",
            "updatedAt": 1762687490000,
            "change": "added"
          },
          {
            "title": "vllm等推理框架的优化",
            "path": "llm/theory/vllm等推理框架的优化.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/vllm%E7%AD%89%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6%E7%9A%84%E4%BC%98%E5%8C%96.md",
            "updatedAt": 1764154018000,
            "change": "added"
          },
          {
            "title": "上下文工程",
            "path": "llm/theory/上下文工程.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B.md",
            "updatedAt": 1763439137000,
            "change": "added"
          },
          {
            "title": "多模态大模型",
            "path": "llm/theory/多模态大模型.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/%E5%A4%9A%E6%A8%A1%E6%80%81%E5%A4%A7%E6%A8%A1%E5%9E%8B.md",
            "updatedAt": 1776345955000,
            "change": "added"
          },
          {
            "title": "大模型基础",
            "path": "llm/theory/大模型基础.md",
            "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80.md",
            "updatedAt": 1762758848000,
            "change": "added"
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
        "url": "https://github.com/golitter/glm-blogs/blob/master/ml/pytorch%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83.md",
        "updatedAt": 1766822360000,
        "change": "added"
      },
      {
        "title": "XGBoost",
        "path": "ml/XGBoost.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/ml/XGBoost.md",
        "updatedAt": 1764568904000,
        "change": "added"
      },
      {
        "title": "机器学习的评价指标",
        "path": "ml/机器学习的评价指标.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/ml/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84%E8%AF%84%E4%BB%B7%E6%8C%87%E6%A0%87.md",
        "updatedAt": 1766822360000,
        "change": "modified"
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
        "url": "https://github.com/golitter/glm-blogs/blob/master/others/golemon-blogs%E9%A1%B5%E9%9D%A2.md",
        "updatedAt": 1783159794000,
        "change": "modified"
      },
      {
        "title": "how read paper",
        "path": "others/how read paper.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/others/how%20read%20paper.md",
        "updatedAt": 1763210695000,
        "change": "added"
      },
      {
        "title": "llm应用相关的简易内容",
        "path": "others/llm应用相关的简易内容.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/others/llm%E5%BA%94%E7%94%A8%E7%9B%B8%E5%85%B3%E7%9A%84%E7%AE%80%E6%98%93%E5%86%85%E5%AE%B9.md",
        "updatedAt": 1763375722000,
        "change": "added"
      }
    ]
  },
  {
    "name": "research",
    "path": "research",
    "count": 1,
    "children": [],
    "files": [
      {
        "title": "deep unfolding",
        "path": "research/deep unfolding.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/research/deep%20unfolding.md",
        "updatedAt": 1783328097000,
        "change": "added"
      }
    ]
  },
  {
    "name": "skills",
    "path": "skills",
    "count": 1,
    "children": [],
    "files": [
      {
        "title": "index",
        "path": "skills/index.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/skills/index.md",
        "updatedAt": 1783158797000,
        "change": "added"
      }
    ]
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
        "url": "https://github.com/golitter/glm-blogs/blob/master/wm/LLM%E5%88%B0World%20Model.md",
        "updatedAt": 1783005765000,
        "change": "added"
      },
      {
        "title": "世界模型入门",
        "path": "wm/世界模型入门.md",
        "url": "https://github.com/golitter/glm-blogs/blob/master/wm/%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8.md",
        "updatedAt": 1783007996000,
        "change": "added"
      }
    ]
  }
] satisfies BlogTreeNode[];
export const recentFiles = [
  {
    "title": "GPT6-Astra分析",
    "path": "llm/agent/GPT6-Astra分析.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/GPT6-Astra%E5%88%86%E6%9E%90.md",
    "date": "2026-09-05 19:02",
    "updatedAt": 1788606121000,
    "change": "added"
  },
  {
    "title": "资料",
    "path": "llm/agent/资料.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/%E8%B5%84%E6%96%99.md",
    "date": "2026-09-04 19:56",
    "updatedAt": 1788522966000,
    "change": "modified"
  },
  {
    "title": "aicoding agent会话成本管控",
    "path": "aicode/aicoding agent会话成本管控.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/aicode/aicoding%20agent%E4%BC%9A%E8%AF%9D%E6%88%90%E6%9C%AC%E7%AE%A1%E6%8E%A7.md",
    "date": "2026-08-17 20:06",
    "updatedAt": 1786968393000,
    "change": "added"
  },
  {
    "title": "J-Space",
    "path": "llm/theory/J-Space.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/llm/theory/J-Space.md",
    "date": "2026-08-17 19:35",
    "updatedAt": 1786966513000,
    "change": "added"
  },
  {
    "title": "deepseek harness",
    "path": "llm/agent/deepseek harness.md",
    "url": "https://github.com/golitter/glm-blogs/blob/master/llm/agent/deepseek%20harness.md",
    "date": "2026-08-16 19:25",
    "updatedAt": 1786879511000,
    "change": "added"
  }
] satisfies RecentFile[];
export const markdownCount = 106;
export const updateTime = "2026-09-09 15:31:36";
