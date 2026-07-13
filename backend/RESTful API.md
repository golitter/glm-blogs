[一文搞懂什么是RESTful API - 知乎](https://zhuanlan.zhihu.com/p/334809573)

https://docs.github.com/en/rest?apiVersion=2026-03-10

REST（英文：Representational State Transfer，简称REST，表现层状态转换）是一种软件架构风格、设计风格，而不是标准，只是提供了一组设计原则和约束条件。主要用于客户端和服务端交互类的软件。

主要特征：

- **以资源为基础**
- **统一接口**：对资源的操作包括获取、创建、修改和删除。使用RESTful风格的接口只能定位其资源，需要具体了解其发生了什么操作动作要从其HTTP请求方法类型上判断。

| HTTP 方法 | 对应操作 | 含义                                             |
| --------- | -------- | ------------------------------------------------ |
| GET       | SELECT   | 从服务器取出资源（一项或多项）                   |
| POST      | CREATE   | 在服务器新建一个资源                             |
| PUT       | UPDATE   | 在服务器更新资源（客户端提供完整资源数据）       |
| PATCH     | UPDATE   | 在服务器更新资源（客户端提供需要修改的资源数据） |
| DELETE    | DELETE   | 从服务器删除资源                                 |

RESTful 和传统 API 大致架构：

![img](https://pic4.zhimg.com/v2-4c87f23be230fdf16dc99398781ebb1b_1440w.jpg)

**URI 指向资源**，意思是：在 RESTful API 设计里，URI 应该表示“资源本身”，而不是“对资源的操作”。

> URI 负责定位资源，HTTP 方法负责表达对资源的操作。

REST架构限制条件：

1. 客户端-服务端：更专注客户端、服务端端分离
2. 无状态：服务端不保存客户端状态，客户端每次请求都携带状态信息
3. 可缓存性：服务端需回复是否可以缓存让客户端甄别是否缓存提高效率
4. 统一接口：通过一定原则设计接口降低耦合，简化系统架构。
5. 分层系统：客户端无法直接知道连接到的是终端还是中间设备。
6. 按需代码：允许灵活发送一些看似特殊的代码（js等）

## RESTful API 设计规范

RESTful API 的核心思想是：**URL 用来定位资源，HTTP 方法用来表达动作。**

一个完整 URL 大致由这些部分组成：

```
scheme://host:port/path?query#fragment
```

例如：

```
https://api.example.com:443/v1/users/1001?fields=name,email#profile
```



| 部分     | 示例                | 含义                                   |
| -------- | ------------------- | -------------------------------------- |
| scheme   | `https`             | 使用的协议                             |
| host     | `api.example.com`   | 服务器域名或 IP                        |
| port     | `443`               | 服务端口，HTTP 默认 80，HTTPS 默认 443 |
| path     | `/v1/users/1001`    | 资源路径，RESTful 设计重点             |
| query    | `fields=name,email` | 查询参数，常用于过滤、分页、排序       |
| fragment | `profile`           | 页面锚点，API 中较少使用               |



**URL 命名规则**

RESTful URL 推荐遵循这些规则：

| 规则              | 推荐             | 不推荐           |
| ----------------- | ---------------- | ---------------- |
| 使用小写          | `/users`         | `/Users`         |
| 使用中划线        | `/user-profiles` | `/user_profiles` |
| 使用复数名词      | `/users`         | `/user`          |
| 不在 URL 中写动词 | `GET /users/1`   | `/getUser?id=1`  |
| 不以 `/` 结尾     | `/users/1`       | `/users/1/`      |
| 不使用文件扩展名  | `/users/1`       | `/users/1.json`  |

最重要的一条是：**URL 中尽量只出现名词，不出现动词。**

> 因为动作已经由 HTTP 方法表达了。

**什么时候可以用 action**

严格 RESTful 里，URL 不推荐出现动词。但实际业务中，有些操作不是简单的增删改查，比如：

```
POST /v1/orders/1001/cancel
POST /v1/users/1001/activate
POST /v1/articles/20/publish
```

这些属于“资源上的业务动作”。如果实在无法用标准 CRUD 表达，可以在资源后面加 action。

**HTTP 动词**

RESTful 中，HTTP 方法本身有语义：

| HTTP 方法 | 语义         | 示例              |
| --------- | ------------ | ----------------- |
| `GET`     | 查询资源     | `GET /users`      |
| `POST`    | 创建资源     | `POST /users`     |
| `PUT`     | 完整更新资源 | `PUT /users/1`    |
| `PATCH`   | 局部更新资源 | `PATCH /users/1`  |
| `DELETE`  | 删除资源     | `DELETE /users/1` |

> `PUT` 和 `PATCH` 的区别尤其重要。
>
> `PUT` 是完整替换。客户端应该提交完整资源数据。

**安全性和幂等性**

RESTful 里经常会提到两个概念：**安全性** 和 **幂等性**。

安全性指：请求不会修改服务器资源。

幂等性指：执行一次和执行多次，最终结果一样。

| 方法     | 是否安全 | 是否幂等 | 说明                               |
| -------- | -------- | -------- | ---------------------------------- |
| `GET`    | 是       | 是       | 只查询，不修改资源                 |
| `POST`   | 否       | 否       | 多次提交可能创建多个资源           |
| `PUT`    | 否       | 是       | 多次完整更新，最终结果一样         |
| `PATCH`  | 否       | 不一定   | 取决于更新逻辑                     |
| `DELETE` | 否       | 是       | 删除一次和多次，最终都是资源不存在 |

**状态码设计**

状态码是服务端告诉客户端“这次请求结果如何”的标准方式。

大类如下：

| 状态码 | 类型       | 含义                 |
| ------ | ---------- | -------------------- |
| `1xx`  | 信息       | 请求已接收，继续处理 |
| `2xx`  | 成功       | 请求成功             |
| `3xx`  | 重定向     | 需要进一步操作       |
| `4xx`  | 客户端错误 | 请求有问题           |
| `5xx`  | 服务端错误 | 服务器处理失败       |

常用状态码如下：

| 状态码 | 含义 | 常见场景 |
| ------ | ---- | -------- |
| `200 OK` | 请求成功 | 查询、更新成功并返回数据 |
| `201 Created` | 创建成功 | `POST` 创建资源成功 |
| `204 No Content` | 成功但无响应体 | `DELETE` 删除成功，或更新成功但不返回数据 |
| `400 Bad Request` | 请求参数错误 | 参数格式错误、JSON 格式错误 |
| `401 Unauthorized` | 未认证 | 未登录、token 缺失或 token 无效 |
| `403 Forbidden` | 无权限 | 已认证，但没有访问该资源的权限 |
| `404 Not Found` | 资源不存在 | 请求的用户、文章、订单等不存在 |
| `422 Unprocessable Entity` | 语义校验失败 | 参数格式正确，但业务校验不通过 |
| `500 Internal Server Error` | 服务器内部错误 | 服务端出现未预期异常 |

**返回数据设计**

RESTful 没有强制规定响应体格式，但现代 API 通常返回 JSON。



## 例子

```go
package main

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

type User struct {
	ID    uint   `json:"id" gorm:"primaryKey"`
	Name  string `json:"name"`
	Email string `json:"email" gorm:"unique"`
	Age   *int    `json:"age"`
}

type CreateUserRequest struct {
	Name  string `json:"name" binding:"required"`
	Email string `json:"email" binding:"required,email"`
	Age   *int    `json:"age"`
}

type UpdateUserRequest struct {
	Name  string `json:"name"`
	Email string `json:"email"`
	Age   *int    `json:"age"`
}

var db *gorm.DB

func main() {
  dsn := "__USER__:__PASSWORD__@tcp(__IP__:__PORT__)/demo?charset=utf8mb4&parseTime=True&loc=Local"

	var err error
	db, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		panic(err)
	}

	db.AutoMigrate(&User{})

	r := gin.Default()

	v1 := r.Group("/v1")
	{
		v1.GET("/users", listUsers)
		v1.GET("/users/:id", getUser)
		v1.POST("/users", createUser)
		v1.PUT("/users/:id", updateUser)
		v1.DELETE("/users/:id", deleteUser)
	}

	r.Run(":8080")
}

func listUsers(c *gin.Context) {
	var users []User

	if err := db.Find(&users).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "查询用户失败"})
		return
	}

	c.JSON(http.StatusOK, users)
}

func getUser(c *gin.Context) {
	var user User

	if err := db.First(&user, c.Param("id")).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"message": "用户不存在"})
		return
	}

	c.JSON(http.StatusOK, user)
}

func createUser(c *gin.Context) {
	var req CreateUserRequest

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}

	user := User{
		Name:  req.Name,
		Email: req.Email,
		Age:   req.Age,
	}

	if err := db.Create(&user).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "创建用户失败"})
		return
	}

	c.JSON(http.StatusCreated, user)
}

func updateUser(c *gin.Context) {
	var user User

	if err := db.First(&user, c.Param("id")).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"message": "用户不存在"})
		return
	}

	var req UpdateUserRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"message": err.Error()})
		return
	}

	user.Name = req.Name
	user.Email = req.Email
	if req.Age != nil {
		user.Age = req.Age
	}

	if err := db.Save(&user).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "更新用户失败"})
		return
	}

	c.JSON(http.StatusOK, user)
}

func deleteUser(c *gin.Context) {
	var user User

	if err := db.First(&user, c.Param("id")).Error; err != nil {
		c.JSON(http.StatusNotFound, gin.H{"message": "用户不存在"})
		return
	}

	if err := db.Delete(&user).Error; err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"message": "删除用户失败"})
		return
	}

	c.Status(http.StatusNoContent)
}
```

```shell
go get github.com/gin-gonic/gin
go get gorm.io/gorm
go get gorm.io/driver/mysql
```





![image-20260713153845318](./RESTful%20API.assets/image-20260713153845318.png)

![image-20260713153921269](./RESTful%20API.assets/image-20260713153921269.png)

![image-20260713153939619](./RESTful%20API.assets/image-20260713153939619.png)

![image-20260713154055101](./RESTful%20API.assets/image-20260713154055101.png)
