https://zhuanlan.zhihu.com/p/410246502

```go
package main

import (
    "net/http"

    "github.com/gin-gonic/gin"
)

func main() {
    r := gin.Default()

    // query 参数示例
    r.GET("/test", func(c *gin.Context) {
        // 1) 有默认值的写法：c.DefaultQuery
        username := c.DefaultQuery("username", "张三")
        // 2) 没有默认值，取不到就是 ""：c.Query
        password := c.Query("password")

        c.JSON(http.StatusOK, gin.H{
            "message":  "success",
            "username": username,
            "password": password,
        })
    })

    r.Run(":8000")
}

```

是从?xxx=yyy 中取值

`c.JSON()`：

```go
func (c *Context) JSON(code int, obj any) {
	c.Render(code, render.JSON{Data: obj})
}
```

`gin.H`：

```go
type H map[string]interface{}
```

通常用于快速构造 JSON 响应（或任意结构的数据）



- `c.DefaultQuery("query", "default_value")`：查询`query`参数，默认为`default_value`。

- `c.Query("query")`：同上，默认为`""`。

- `c.Request.URL.Query()`：获取所有的query参数，返回`url.Values`（`map[string][string]`）。每个key可能对应多个values

  ```go
  
  		queryParams := c.Request.URL.Query()
  		// 3) 取到所有的 query 参数：c.Request.URL.Query()
  		fmt.Println("queryParams:", queryParams, "type:", reflect.TypeOf(queryParams)) // url.Values
  
  		for key, values := range queryParams {
  			for _, value := range values {
  				fmt.Printf("key: %s, value: %s\n", key, value)
  			} // 输出 string类型
  		}
  ```



```go
package main

import (
    "net/http"

    "github.com/gin-gonic/gin"
)

type User struct {
    UserName string `json:"username"` // 指定 JSON 字段名
    PassWord string `json:"password"`
}

func main() {
    r := gin.Default()

    r.POST("/user/info", func(c *gin.Context) {
        var userBody User
        // ShouldBindJSON 会检查 Content-Type 是否为 JSON，并解码
        if err := c.ShouldBindJSON(&userBody); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{
                "err_no":  400,
                "message": "Post Data Err",
            })
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "username": userBody.UserName,
            "password": userBody.PassWord,
        })
    })

    r.Run(":8000")
}
```

`ShouldBindJSON`**只能调用一次**，会检查`Content-Type`是否为json，并解码。

- 读取原始JSON字符串

  ```go
  rawData, err := c.GetRawData()
  if err != nil {
      // 处理错误
  }
  // rawData 是 []byte，你可以自己用 json.Unmarshal 解码
  var body map[string]interface{}
  json.Unmarshal(rawData, &body)
  ```



```go
package main

import (
    "net/http"
    "github.com/gin-gonic/gin"
)

// 绑定路径参数
type PathParams struct {
    Username string `uri:"username" binding:"required"`
    Password string `uri:"password" binding:"required"`
}

// 绑定 JSON Body
type JSONBody struct {
    Email string `json:"email" binding:"required"`
    Age   int    `json:"age"`
}

func main() {
    r := gin.Default()

    r.POST("/user/info/:username/:password", func(c *gin.Context) {
        var path PathParams
        if err := c.ShouldBindUri(&path); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": "path error: " + err.Error()})
            return
        }

        var body JSONBody
        if err := c.ShouldBindJSON(&body); err != nil {
            c.JSON(http.StatusBadRequest, gin.H{"error": "json error: " + err.Error()})
            return
        }

        c.JSON(http.StatusOK, gin.H{
            "path_username": path.Username,
            "path_password": path.Password,
            "json_email":    body.Email,
            "json_age":      body.Age,
        })
    })

    r.Run(":8000")
}
```

`ShouldBindUri`，可以直接把` :xxx `绑定到结构体，并配合 `binding `标签做校验。

`ShouldBindJSON`绑定body传入的JSON数据。

