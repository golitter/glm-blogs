https://github.com/colbymchenry/codegraph

https://zhuanlan.zhihu.com/p/2043160358018348348

codegraph是给aicoding做前置索引的工具。前置索引我理解的就是代码库中的**静态结构信息**，这些内容很多时候aicoding每次都要进行检索，在大型项目中往往会浪费不少token。

它的实现细节是用**tree-sitter**，将源代码转为语法树AST。再每一次代码更改时，只计算有影响的结构；语法错误、也可以返回尽可能正确的结构。

> 与其对应的还有LSP（语言服务器协议）
>
> tree-sitter比LSP弱，但是响应快，不需要项目能编译，有一份全平台的wasm，增量修改快。

它的存储采用的是sqlite+fts5，主要通过符号名+fst5+图遍历完成。



