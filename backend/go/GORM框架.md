https://mp.weixin.qq.com/s/plzG1mCK8yZwVQOSKZi2XQ

https://gorm.io/zh_CN/docs/index.html

https://blog.csdn.net/u012955829/article/details/142289384



ORM（Object-Relational Mapping，对象关系映射），就像是一位翻译官，在面向对象的编程语言和关系型数据库之间进行翻译。允许使用面向对象的方式来操作数据库，将数据库表映射到编程语言中的类，将表中的记忆映射到类的实例，以及将表的字段映射到对象的属性。	



```go
package hello

import (
	"gorm.io/driver/mysql"  // GORM 的 MySQL 驱动
	"gorm.io/gorm"          // GORM ORM 框架核心包
	"sync"                  // 同步工具包，用于实现单例模式
)

var (
	// db 全局数据库连接实例，通过单例模式保证只初始化一次
	db *gorm.DB
	// dbOnce 用于保证并发安全下的单例初始化，确保 getDB 只执行一次连接创建
	dbOnce sync.Once
	// dsn MySQL 数据源连接字符串（Data Source Name）
	// 格式：用户名:密码@(主机:端口)/数据库名?参数
	// timeout：连接超时时间
	// readTimeout：读操作超时时间
	// writeTimeout：写操作超时时间
	// charset：字符集编码，utf8mb4 支持完整的 Unicode（包括 emoji）
	// parseTime：自动将 MySQL 的 TIME/DATE/DATETIME 类型解析为 Go 的 time.Time
	// loc：时区设置为本地时区
	dsn = "username:password@(ip:port)/database?timeout=5000ms&readTimeout=5000ms&writeTimeout=5000ms&charset=utf8mb4&parseTime=true&loc=Local"
)

// getDB 获取全局数据库连接实例（单例模式）
// 首次调用时会创建连接，后续调用直接返回已创建的实例
// 返回值：
//   - *gorm.DB：数据库连接实例
//   - error：连接创建过程中的错误信息
func getDB() (*gorm.DB, error) {
	var err error
	// Do 中的函数只会执行一次，无论有多少 goroutine 并发调用 getDB
	// 从而保证数据库连接只被初始化一次，避免重复创建连接
	dbOnce.Do(func() {
		// 使用 GORM 打开 MySQL 连接
		// mysql.Open(dsn) 传入 DSN 创建 MySQL 驱动实例
		// &gorm.Config{} 使用默认配置（可按需自定义日志、命名策略等）
		db, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
	})
	// 如果初始化失败，db 为 nil，err 不为 nil
	// 调用方需要检查 err 来判断连接是否可用
	return db, err
}
```



## 持久化对象 PO

GORM 提供了 `gorm.Model`：

```go
type Model struct {
    // 主键 id
    ID        uint `gorm:"primarykey"`
    // 创建时间
    CreatedAt time.Time
    // 更新时间
    UpdatedAt time.Time
    // 删除时间
    DeletedAt DeletedAt `gorm:"index"`
}
```

通过引入`gorm.Model`可以为PO添加这四个字段。

```go
type PO struct {
    gorm.Model
}
```

当添加了`deleteAt`字段，则默认开启**软删除**，在执行删除时不会立即删除数据，而是仅仅将po的`deleteAt`字段设置为非空。

可以通过覆盖`deleteAt`字段进行去除该字段：

```go
type User struct {
	gorm.Model // 依然组合 gorm.Model
	
	// 使用 gorm:"-" 忽略该字段，数据库建表时不会创建 deleted_at 列
	// 这样就变相屏蔽了 gorm.Model 中的 DeletedAt 软删除功能
	DeletedAt interface{} `gorm:"-"` 
```

> 这个"字段遮蔽"技巧在 GORM v2 中已经不可靠了。GORM v2 会通过反射遍历嵌入结构体的所有字段，即使你在 `User` 中用同名 `DeletedAt` 覆盖，GORM 内部仍能从 `gorm.Model` 中识别出 `DeletedAt gorm.DeletedAt` 并启用软删除。
>
> 可靠的做法是：**不嵌入 `gorm.Model`，手动定义需要的字段**。

通过下面方式对表命名：

为`User`表结构进行命名为`t_users`。

```go
func (User) TableName() string {
	return "t_users"
}
```



之后可以根据go程序中定义的表结构来进行创建表：

```go
	// 自动建表（根据 User 结构体和标签）
	db.AutoMigrate(&User{})
```

建表不是只建一次，而是**每次应用启动都会执行检查，但是其更智能、安全的增量更新**。

- 表不存在时：创建表
- 表已存在且结构没有发生变化：什么都不做
- 表已存在但是新添加字段：自动加列

`AutoMigrate`**不会删除列，不会修改列的类型/约束**。





```go
package main

import (
	"database/sql"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"sync"
	"time"
)


// User 持久化对象(PO)，对应数据库中的 user 表
type User struct {
	// 2.1 组合 gorm.Model
	// 包含了 ID(uint主键)、CreatedAt、UpdatedAt、DeletedAt
	// 启用了 DeletedAt，后续执行删除时会自动开启软删除机制
	// 不嵌入 gorm.Model，手动定义字段以排除 DeletedAt，从而禁用软删除
	ID        uint           `gorm:"primarykey"`
	CreatedAt time.Time
	UpdatedAt time.Time

	// 2.2 使用标签精细控制数据库映射
	// column:指定列名为name；type:指定为varchar(15)；unique_index:设为唯一索引；not null:非空
	Name string `gorm:"column:name;type:varchar(15);unique_index;not null"` 

	// 2.3 零值问题解决方案一：使用指针类型
	// 默认值设为 18。如果使用 int，传入 0 时 GORM 会忽略；
	// 使用 *int，只要指针非空(即使指向0)，GORM 就会明确将 0 写入数据库
	Age *int `gorm:"column:age;default:18"` 

	// 2.3 零值问题解决方案二：使用 sql.NullXX 类型
	// 比如 sql.NullInt64、sql.NullString 等
	// Valid 为 true 时，代表显式赋值；Valid 为 false 时，代表未赋值(对应DB的NULL)
	Score sql.NullInt64 `gorm:"column:score"` 

	// 2.2 自增列标签
	// Num 列的数值会逐行递增
	Num int `gorm:"auto_increment"` 

	// 额外演示：字符串的零值问题
	// 如果希望将用户的昵称更新为空字符串 ""，也需要使用指针或 sql.NullString
	NickName *string `gorm:"column:nickname;type:varchar(50)"` 
}

// 2.5 表名指定方式一：实现 TableName 方法
// 只要 User 结构体实现了这个方法，GORM 在执行迁移、增删改查时，都会使用 "t_user" 作为表名
// 注意：这里接收者是 User 类型即可，不需要指针 *User
func (User) TableName() string {
	return "t_users" // 比如我们强制表名为 t_user，不带复数s
}

var (
	// db 全局数据库连接实例，通过单例模式保证只初始化一次
	db *gorm.DB
	// dbOnce 用于保证并发安全下的单例初始化，确保 getDB 只执行一次连接创建
	dbOnce sync.Once
	// dsn MySQL 数据源连接字符串（Data Source Name）
	// 格式：用户名:密码@(主机:端口)/数据库名?参数
	// timeout：连接超时时间
	// readTimeout：读操作超时时间
	// writeTimeout：写操作超时时间
	// charset：字符集编码，utf8mb4 支持完整的 Unicode（包括 emoji）
	// parseTime：自动将 MySQL 的 TIME/DATE/DATETIME 类型解析为 Go 的 time.Time
	// loc：时区设置为本地时区
	dsn = "root:123456@(localhost:3306)/sql_learn?timeout=5000ms&readTimeout=5000ms&writeTimeout=5000ms&charset=utf8mb4&parseTime=true&loc=Local"
)

// getDB 获取全局数据库连接实例（单例模式）
// 首次调用时会创建连接，后续调用直接返回已创建的实例
// 返回值：
//   - *gorm.DB：数据库连接实例
//   - error：连接创建过程中的错误信息
func getDB() (*gorm.DB, error) {
	var err error
	// Do 中的函数只会执行一次，无论有多少 goroutine 并发调用 getDB
	// 从而保证数据库连接只被初始化一次，避免重复创建连接
	dbOnce.Do(func() {
		// 使用 GORM 打开 MySQL 连接
		// mysql.Open(dsn) 传入 DSN 创建 MySQL 驱动实例
		// &gorm.Config{} 使用默认配置（可按需自定义日志、命名策略等）
		db, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
	})
	// 如果初始化失败，db 为 nil，err 不为 nil
	// 调用方需要检查 err 来判断连接是否可用
	return db, err
}

```

```go
package main

import (
	"fmt"
)
func main() {
	db, err := getDB()
	if err != nil {
		panic("Failed to connect to database: " + err.Error())
	}
	// 连接成功，db 可用于后续的数据库操作

	// 自动建表（根据 User 结构体和标签）
	db.AutoMigrate(&User{})

	fmt.Println("建表完成")
}
```

