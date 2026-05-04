[gin框架底层技术原理剖析_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1zm4y177mb?spm_id_from=333.788.videopod.episodes&vd_source=13dfbe5ed2deada83969fafa995ccff6)

[解析 Gin 框架底层原理](https://mp.weixin.qq.com/s/x8i9HvAzIHNbHCryLw5icg)

Gin框架

- 支持中间件操作
- 更方便使用
- 更强大的路由解析能力

安装方式：

```shell
go get github.com/gin-gonic/gin@latest
```



gin框架是在net/http标准库下提供了gin.Engine对象作为Handier注入其中，从而实现路由注册/匹配、请求处理链路的优化。

Gin框架使用示例

- 构造gin.Engine实例：gin.Default()
- 路由组注册中间件：Engine.Use()
- 路由组注册POST方法下的handler：Engine.POST()
- 启动http server：Engine.Run()

```go
package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
   r := gin.Default() // 创建一个默认的Gin Engine，本质是 http Handler
   r.Use(func(c *gin.Context) { // 定义一个中间件函数，打印请求路径
      path := c.Request.URL.Path
      println("请求路径:", path)
      c.Next() // 调用下一个处理器
   })

   r.POST("/ping", func(c *gin.Context) { // 定义一个POST请求的处理器，路径为 /ping
      c.JSON(http.StatusOK, "pong")
   })

   if err := r.Run(":8080"); err != nil { // 启动HTTP服务器，监听8080端口
      panic(err)
   }
   
}
```

## gin.Engine

gin.Engine是一个宏观的http Handler。里面包含三个：pool sync.Pool、RouterGroup、trees methodTrees。

- pool sync.Pool复用一系列gin.Context工具类。如果要用gin.Context，会优先去回收站中复用gin.Context，并将这个gin.Context的业务数据清除。
- RouterGroup（路由组），针对每一个请求有一个路径处理函数。
- tree methodTrees，根据路由方法有不同的树（压缩前缀树）。

Engine是Gin中构建的http Handler，实现了net/http标准库包下Handler interface的抽象方法：Handler.ServeHTTP，**因此，可以作为Handler注入到net/http的Server中。

Engine包含核心内容：

```go
type Engine struct {
	// 路由组
	RouterGroup
	
	// context 对象池
	pool sync.Pool
	
	// 方法树
	trees methodTrees
}
```

**RouterGroup**

```go
type RouterGroup struct {
	Handlers HandlersChain
	basePath string
	engine *Engine
	root bool
}
```

- Handlers：路由组共同的handler处理函数链。拼接公共的handlers和自己的handlers作为最终的handers链
- basePath：路由组的基础路径，组下的节点将拼接RouterGroup的bashPath和自己的path组成最终的absolutePath
- engine：指向路由组从属的Engine
- root：标识路由组是否位于Engine的根节点

**HandlersChain**

```go
type handlersChain []HandlerFunc
type HandletFunc func(*Context)
```

HandlersChain是由多个路由处理函数HandlerFunc构成的处理函数链，在使用时会按照索引的先后顺序依次调用HandlerFunc。

## handler流程

创建Engine，注册中间件 ，具体路径逻辑。

创建Engine：

1. 创建了一个gin.Engine实例
2. 创建Engine的首个RouterGroup，对应的处理函数链Handlers为nil，基础路径basePath为"/"，root标识为true
3. 构造了9棵方法路由树，对应9中http方法
4. 创建gin.Context对象池

通过Engine.Use方法实现中间件的注册，会将注册的中间件操作添加到RouterGroup.Handlers中。后续RouterGroup下新注册的handler都会在前缀中拼接上这部分路由组公共的handlers

以http POST为例子，注册handler方法调用顺序为RouterGroup.POST -> RouterGroup.handle，接下来完成三个操作：

1. 拼接出absolutePath
2. 拼接出完成handlers链
3. 以absoluterPath和handlers组成kv对添加到路由树中

> **外层先于内层，父级先于子级，同层看书写顺序，最后才是真业务。**
>
> > 来自chatglm
>
> ```go
> r := gin.New()
> 
> // 1. Engine 根节点注册中间件 A
> r.Use(MiddlewareA)
> 
> // 2. 第一层路由组 /v1 注册中间件 B
> v1 := r.Group("/v1")
> v1.Use(MiddlewareB)
> 
> // 3. 第二层路由组 /v1/user 注册中间件 C
> user := v1.Group("/user")
> user.Use(MiddlewareC)
> 
> // 4. 具体的路由注册业务逻辑 D
> user.GET("/profile", HandlerD)
> 
> ```
>
> ```text
> 请求进入
>   │
>   ▼
> 【Middleware A】 开始执行
>   │ 调用 c.Next()
>   ▼
> 【Middleware B】 开始执行
>   │ 调用 c.Next()
>   ▼
> 【Middleware C】 开始执行
>   │ 调用 c.Next()
>   ▼
> 【Handler D】   真正的业务逻辑处理，返回响应
>   │
>   ▼ (D 执行完毕，返回到 C 的 c.Next() 之后)
> 【Middleware C】 结束后续逻辑
>   │
>   ▼ (返回到 B 的 c.Next() 之后)
> 【Middleware B】 结束后续逻辑
>   │
>   ▼ (返回到 A 的 c.Next() 之后)
> 【Middleware A】 结束后续逻辑
>   │
>   ▼
> 请求结束
> 
> ```



## Gin服务启动流程

启动Engine.Run方法后，会将gin.Engine本身作为net/http包下Handler interface的实现类，并调用http.ListenAndServe方法启动服务

> net/http标准库的http.ListenAndServe会基于主动轮询＋IO多路复用运行，因此程序运行时会始终阻塞Engine.Run方法。

服务端接收到http请求时，会通过Handler.ServeHTTP方法进行处理（此时是gin.Engine），处理请求的核心步骤：

1. 对于每个http请求，会为其分配一个gin.Context，在handlers链路中持续向下传递
2. 调用Engine.handleHTTPRequest方法，从路由树中获取handlers链，然后遍历调用
3. 处理完http请求后，会将gin.Context进行回收

Engine.handleHTTPRquest方法步骤：

1. 根据http method取得对应的methodTree
2. 根据path从methodTree中找到对应的handlers链
3. 将handlers链注入到gin.Context中，通过Context.Next方法按照顺序遍历调用handler



## 路由树

> 压缩前缀树，前缀树改良版本，优化点主要在于空间的节省：如果某个子节点是其父节点的**唯一孩子**，则与父节点进行合并。

- path匹配时不是完全精确匹配，比如末尾'/'符号的增减，全符号'*'的处理（`user/\*`，map无法胜任。例如`/ping`和`/ping/`本质是一样的。
- path串通常存在基于分组分类的公共前缀，适合使用前缀树进行管理，可以节省存储空间。

