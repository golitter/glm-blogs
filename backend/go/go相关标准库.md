## encoding/json

> **在 Go 中，如果想要对结构体进行序列化与反序列化，字段必须是对外暴露的，即首字母大写。**
> 因为小写字母开头的字段在 Go 中是私有的，外部的包（包括 `encoding/json` 等）无法读取和修改它们。

- `Marshal`：序列化，将go对象转为json字符串
- `UnMarshal`：反序列化，把json转为go对象
- `MarshalIndent`：带缩进的序列化，方便调试阅读

```go
func Marshal(v any) ([]byte, error) //xml序列化

func MarshalIndent(v any, prefix, indent string) ([]byte, error) //格式化

func Unmarshal(data []byte, v any) error //反序列化
```

```go
package main

import (
  "fmt"
  "encoding/json"
)
type User struct {
	ID       int    `json:"id"`               // 基础用法：字段重命名
	Username string `json:"username"`         // 基础用法：字段重命名
	Password string `json:"-"`                // 高级用法：`-` 表示忽略该字段，序列化和反序列化都会跳过它
	NickName string `json:"nickname,omitempty"` // 高级用法：omitempty 表示如果该字段为空值(零值)，则在 JSON 中省略该字段
	Email    string `json:"email,omitempty"`  // 空值包括：空字符串、数字0、false、空指针、空切片等
	Profile  Profile `json:"profile"`         // 嵌套结构体
}
type Profile struct {
	Age int `json:"age"`
}
func main() {
	// 实例化一个用户，故意不设置 NickName 和 Email 来测试 omitempty
	user := User{
		ID:       1001,
		Username: "jack",
		Password: "super_secret_123", // 这个字段不会出现在 JSON 中
		NickName: "",                 // 空字符串
		Email:    "",                 // 空字符串
		Profile: Profile{
			Age: 25,
		},
	}

  jsonBytes, err := json.Marshal(user)
  if err != nil {
    fmt.Println(err)
    return 
  }
	fmt.Println("【普通序列化】适合传给前端:")
	fmt.Println(string(jsonBytes))

  // 使用 json.MarshalIndent 生成格式化的 JSON（带缩进，适合人类阅读/写文件）
	jsonPrettyBytes, err := json.MarshalIndent(user, "", "  ") // 前缀为空，缩进为两个空格
	if err != nil {
		fmt.Println("格式化序列化失败:", err)
		return
	}
	fmt.Println("\n【格式化序列化】适合打印日志或存入配置文件:")
	fmt.Println(string(jsonPrettyBytes))

// 模拟从网络接收到的 JSON 字符串
	jsonStr := `{"id":1002,"username":"tom","nickname":"汤姆","profile":{"age":30}}`
	
	var user2 User
	// 注意：传入的是字符串的切片 []byte，以及结构体的指针 &user2
    // 第二个参数必须传入指针，不然是副本，外部得不到哦
	err = json.Unmarshal([]byte(jsonStr), &user2)
	if err != nil {
		fmt.Println("反序列化失败:", err)
		return
	}
	
	fmt.Println("\n【反序列化结果】:")
	fmt.Printf("解析出来的 Go 对象: %+v\n", user2)
	fmt.Printf("提取密码字段: %s (即使 JSON 里没有，Go 也会有默认零值)\n", user2.Password)


}
```



## net/http

```go
package main

import (
  "fmt"
  "io"
  "net/http"
)

func main() {
  res, err := http.Get("https://www.baidu.com")
  if err != nil {
    fmt.Println("Error:", err)
    return
  }

  // defer 保证即使后续代码发生 panic，也能执行 Close。
  defer res.Body.Close()

  // 使用 io.ReadAll 从 resp.Body 一次性读取全部响应内容。
  body, err := io.ReadAll(res.Body)
  if err != nil {
    fmt.Println("Error reading response body:", err)
    return
  }
  fmt.Println(string(body))
}
```



```go
type Person struct {
   UserId   string
   Username string
   Age      int
   Address  string
}

func main() {
   person := Person{
      UserId:   "120",
      Username: "jack",
      Age:      18,
      Address:  "usa",
   }

   json, _ := json.Marshal(person) // 将person对象转换为json格式的
   reader := bytes.NewReader(json) // 将json格式的字符串转换为io.Reader接口类型的对象

   resp, err := http.Post("https://golang.org", "application/json;charset=utf-8", reader)
   if err != nil {
      fmt.Println(err)
   }
   defer resp.Body.Close()
}
```

配置一个客户端来达到更加细致化的需求

```go
func main() {
  client := &http.Client{} // 创建一个HTTP客户端
  request, _ := http.NewRequest("GET", "https://golang.org", nil) // 创建一个HTTP GET请求
  resp, _ := client.Do(request) // 发送请求并获取响应
  defer resp.Body.Close() // 确保在函数结束时关闭响应体
}
```

```go
func main() {

   // 当请求的是 ~/index 时，会调用这个函数
   http.HandleFunc("/index", func(responseWriter http.ResponseWriter, request *http.Request) {
      fmt.Println(responseWriter, "index")
   })
   http.ListenAndServe(":8080", nil)
}
```

