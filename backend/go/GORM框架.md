# GORM 框架

https://mp.weixin.qq.com/s/plzG1mCK8yZwVQOSKZi2XQ

https://gorm.io/zh_CN/docs/index.html

https://blog.csdn.net/u012955829/article/details/142289384

GORM 是 Go 语言常用的 ORM 框架。ORM 即 Object-Relational Mapping，对象关系映射。

- 结构体对应数据库表
- 结构体字段对应表字段
- 结构体实例对应表中的一行数据

## 安装

如果项目还没有 `go.mod`，先初始化：

```bash
go mod init gorm-demo
```

安装 GORM：

```bash
go get -u gorm.io/gorm
```

如果使用 MySQL，还需要安装 MySQL 驱动：

```bash
go get -u gorm.io/driver/mysql
```

代码里导入：

```go
import (
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)
```

## 连接 MySQL

```go
package main

import (
	"time"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

func initDB() (*gorm.DB, error) {
	dsn := "root:mysql@(localhost:3306)/abc?charset=utf8mb4&parseTime=true&loc=Local"

	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, err
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, err
	}

	sqlDB.SetMaxOpenConns(20)
	sqlDB.SetMaxIdleConns(10)
	sqlDB.SetConnMaxLifetime(time.Hour)

	return db, nil
}
```

DSN 参数：

- `charset=utf8mb4`：支持完整 Unicode。
- `parseTime=true`：把 MySQL 时间类型解析为 Go 的 `time.Time`。
- `loc=Local`：使用本地时区。

连接池配置：

```go
sqlDB.SetMaxOpenConns(20)
sqlDB.SetMaxIdleConns(10)
sqlDB.SetConnMaxLifetime(time.Hour)
```

- `SetMaxOpenConns`：最大打开连接数。
- `SetMaxIdleConns`：最大空闲连接数。
- `SetConnMaxLifetime`：连接最大存活时间。

## 模型定义

```go
type User struct {
	ID        uint           `gorm:"primaryKey"`
	Name      string         `gorm:"column:name;type:varchar(32);not null"`
	Email     string         `gorm:"column:email;type:varchar(100);uniqueIndex;not null"`
	Age       *int           `gorm:"column:age;type:int;default:18"`
	Birthday  *time.Time     `gorm:"column:birthday;type:date"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}
```

`ID uint primaryKey` 默认会映射成 MySQL 自增主键。

```go
ID uint `gorm:"primaryKey"`
```

大致对应：

```sql
id bigint unsigned AUTO_INCREMENT PRIMARY KEY
```

如果不想自增：

```go
ID uint `gorm:"primaryKey;autoIncrement:false"`
```

字段 tag 外面的空格只是 Go 代码格式，`gofmt` 会自动处理。

```go
Email     string         `gorm:"column:email;type:varchar(100);uniqueIndex;not null"`
```

多个 tag 之间可以有空格：

```go
Email string `gorm:"column:email" json:"email"`
```

GORM tag 内部不建议在分号后随意加空格：

```go
// 推荐
`gorm:"column:email;type:varchar(100);uniqueIndex;not null"`

// 不推荐
`gorm:"column:email; type:varchar(100); uniqueIndex; not null"`
```

## 表名

通过 `TableName` 指定表名：

```go
func (User) TableName() string {
	return "l_users"
}
```

完整写法：

```go
func (u User) TableName() string {
	return "l_users"
}
```

如果没有使用接收者变量，可以省略变量名：

```go
func (User) TableName() string {
	return "l_users"
}
```

GORM 会识别类似接口：

```go
type Tabler interface {
	TableName() string
}
```

Go 是隐式实现接口，只要 `User` 有 `TableName() string` 方法即可。

## 自动迁移

```go
err := db.AutoMigrate(&User{})
if err != nil {
	panic(err)
}
```

`AutoMigrate` 会自动创建表、添加新增字段，不会主动删除列；部分列类型或约束变更可能会处理，但复杂变更不要完全依赖它。开发阶段方便，生产环境需要更谨慎。

## gorm.Model

GORM 内置模型：

```go
type Model struct {
	ID        uint `gorm:"primarykey"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}
```

可以直接嵌入：

```go
type User struct {
	gorm.Model
	Name string
}
```

`DeletedAt gorm.DeletedAt` 会开启软删除。

```go
db.Delete(&user)
```

普通删除实际是更新 `deleted_at`：

```sql
UPDATE l_users SET deleted_at = '2026-07-07 12:00:00' WHERE id = 1;
```

普通查询会自动过滤：

```sql
WHERE deleted_at IS NULL
```

物理删除：

```go
db.Unscoped().Delete(&user)
```

对应：

```sql
DELETE FROM l_users WHERE id = 1;
```

查询包含软删除的数据：

```go
db.Unscoped().Find(&users)
```

`DeletedAt` 加索引是因为查询经常带 `deleted_at IS NULL`。但单列索引不一定总是最优，实际业务常用联合索引：

```sql
(email, deleted_at)
(user_id, deleted_at)
```

## time.Time

```go
CreatedAt time.Time
UpdatedAt time.Time
Birthday  *time.Time `gorm:"column:birthday;type:date"`
```

`time.Time` 不能表示 `NULL`，零值是：

```text
0001-01-01 00:00:00 +0000 UTC
```

允许为空一般使用：

```go
Birthday *time.Time
```

或者：

```go
Birthday sql.NullTime
```

多个时间字段：

```go
type User struct {
	CreatedAt time.Time  // GORM 自动维护创建时间
	UpdatedAt time.Time  // GORM 自动维护更新时间
	Birthday  *time.Time `gorm:"type:date"`
	LoginAt   *time.Time `gorm:"type:datetime"`
	ExpireAt  *time.Time `gorm:"type:datetime(3)"`
}
```

- `date`：只存日期。
- `datetime`：日期和时间。
- `datetime(3)`：保留毫秒。

显示格式在输出时处理：

```go
fmt.Println(user.Birthday.Format("2006-01-02"))
fmt.Println(user.LoginAt.Format("2006-01-02 15:04:05"))
```

Go 时间格式模板：

```text
2006-01-02 15:04:05
```

## 指针字段

需要区分“没传/NULL”和“传了零值”时，用指针。

```go
Age int
```

普通 `int` 不能区分没传和 `0`。

```go
Age *int
```

可以区分：

```text
Age == nil  -> 没传 / NULL / 走默认值
*Age == 0   -> 明确传入 0
*Age == 18  -> 明确传入 18
```

常见指针字段：

```go
Age      *int
NickName *string
Birthday *time.Time
ExpireAt *time.Time
```

常见非指针字段：

```go
ID        uint
Name      string
CreatedAt time.Time
UpdatedAt time.Time
```

## 默认值

```go
Age *int `gorm:"column:age;type:int;default:18"`
```

创建时 `Age == nil`，通常不插入 `age` 字段，让数据库默认值生效。

```go
user := User{
	Name:  "Tom",
	Email: "tom@example.com",
	Age:   nil,
}

db.Create(&user)
```

大致 SQL：

```sql
INSERT INTO l_users (name, email) VALUES ('Tom', 'tom@example.com');
```

数据库结果：

```text
age = 18
```

明确传入 `0`：

```go
age := 0

user := User{
	Name:  "Baby",
	Email: "baby@example.com",
	Age:   &age,
}

db.Create(&user)
```

数据库结果：

```text
age = 0
```

`default:18` 是数据库默认值，不是 Go 自动把 `nil` 改成 `18`。

## 创建

```go
func createUser(db *gorm.DB) {
	user := User{
		Name:  "John Doe",
		Email: "john.doe@example.com",
	}

	result := db.Create(&user)
	if result.Error != nil {
		fmt.Println("Error creating user:", result.Error)
		return
	}

	fmt.Println("Create Successful, ID:", user.ID)
}
```

创建成功后，自增 ID 会回填到 `user.ID`。

## 查询

```go
func queryUser(db *gorm.DB) {
	var user User

	err := db.Where("email = ?", "john.doe@example.com").First(&user).Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			fmt.Println("用户不存在")
			return
		}

		fmt.Println("查询用户失败:", err)
		return
	}

	fmt.Println("查询用户成功:", user)
}
```

链式调用可以拆开理解：

```go
query := db.Where("email = ?", "john.doe@example.com")
result := query.First(&user)
err := result.Error
```

- `Where`：查询条件。
- `First(&user)`：执行查询，并把结果写入 `user`。
- `.Error`：取错误。

由于有软删除字段，普通查询大致是：

```sql
SELECT * FROM l_users
WHERE email = 'john.doe@example.com'
  AND deleted_at IS NULL
ORDER BY id
LIMIT 1;
```

指针字段打印时可能是地址，需要解引用：

```go
if user.Age != nil {
	fmt.Println("Age:", *user.Age)
} else {
	fmt.Println("Age: nil")
}

if user.Birthday != nil {
	fmt.Println("Birthday:", user.Birthday.Format("2006-01-02"))
} else {
	fmt.Println("Birthday: nil")
}
```

`Age` 有默认值，插入后通常不是 nil；`Birthday` 没有默认值，没传就是 NULL，查询回来就是 nil。

## 更新

查询时 `First(&user)` 已经能通过 `&user` 知道模型；更新 map 时只知道字段，不知道表，所以要指定 `Model`。

```go
var user User
db.Where("email = ?", "tom@example.com").First(&user)
```

```go
db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Updates(map[string]any{
		"name": "Jerry",
		"age":  22,
	})
```

单字段更新：

```go
err := db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Update("age", 0).Error
```

多字段更新推荐 map，map 会按 key 更新，零值也会更新：

```go
err := db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Updates(map[string]any{
		"name": "",
		"age":  0,
	}).Error
```

虽然模型里 `Age` 是 `*int`，但 map 更新写的是数据库列值，`"age": 22` 或 `"age": 0` 可以直接用 int。

结构体更新默认跳过零值：

```go
err := db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Updates(User{
		Name: "Jerry",
	}).Error
```

结构体更新需要匹配字段类型，如果 `Age` 是 `*int`，要写 `Age: &age`。

常见零值：

```text
int       -> 0
string    -> ""
bool      -> false
pointer   -> nil
time.Time -> 0001-01-01 00:00:00
```

例子：

```go
db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Updates(User{
		Name: "",
	})
```

`Name` 是空字符串，默认会被跳过。

指针非 nil 时，即使指向 `0`，也会更新：

```go
age := 0

db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Updates(User{
		Age: &age,
	})
```

使用 `Select` 可以让结构体更新零值，例如把 `Name` 更新为空字符串：

```go
db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Select("Name").
	Updates(User{
		Name: "",
	})
```

多个字段：

```go
age := 0

db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Select("Name", "Age").
	Updates(User{
		Name: "",
		Age:  &age,
	})
```

`Select("*")` 会让结构体所有字段参与更新，包括零值，容易误覆盖字段。

```go
age := 0

db.Model(&User{}).
	Where("email = ?", "tom@example.com").
	Select("*").
	Updates(User{
		Age: &age,
	})
```

实际开发更推荐精确字段：

```go
Select("Age")
```

而不是：

```go
Select("*")
```

## 更新 nil 和默认值

```go
db.Model(&User{}).
	Where("email = ?", "john.doe@example.com").
	Updates(map[string]any{
		"age": nil,
	})
```

这是把 `age` 更新为 `NULL`：

```sql
UPDATE l_users SET age = NULL WHERE email = 'john.doe@example.com';
```

不会触发 `default:18`。默认值只在插入时字段缺省才生效。

更新成默认值可以直接写：

```go
db.Model(&User{}).
	Where("email = ?", "john.doe@example.com").
	Update("age", 18)
```

也可以用数据库 `DEFAULT`：

```go
db.Model(&User{}).
	Where("email = ?", "john.doe@example.com").
	Update("age", gorm.Expr("DEFAULT"))
```

## 实际开发建议

单字段用 `Update`：

```go
db.Model(&User{}).
	Where("email = ?", email).
	Update("age", 0)
```

多字段局部更新用 map：

```go
db.Model(&User{}).
	Where("email = ?", email).
	Updates(map[string]any{
		"name": name,
		"age":  age,
	})
```

接口请求使用 DTO 指针字段：

```go
type UpdateUserReq struct {
	Name *string `json:"name"`
	Age  *int    `json:"age"`
}
```

动态组装 map：

```go
func UpdateUser(db *gorm.DB, email string, req UpdateUserReq) error {
	updates := map[string]any{}

	if req.Name != nil {
		updates["name"] = *req.Name
	}

	if req.Age != nil {
		updates["age"] = *req.Age
	}

	if len(updates) == 0 {
		return nil
	}

	return db.Model(&User{}).
		Where("email = ?", email).
		Updates(updates).Error
}
```

这样可以区分：

```json
{}
```

表示什么都不改。

```json
{"age":0}
```

表示明确把年龄改成 0。

```json
{"name":""}
```

表示明确把名字改成空字符串。

## 完整练习代码

```go
package main

import (
	"errors"
	"fmt"
	"time"

	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

type User struct {
	ID        uint           `gorm:"primaryKey"`
	Name      string         `gorm:"column:name;type:varchar(32);not null"`
	Email     string         `gorm:"column:email;type:varchar(100);uniqueIndex;not null"`
	Age       *int           `gorm:"column:age;type:int;default:18"`
	Birthday  *time.Time     `gorm:"column:birthday;type:date"`
	CreatedAt time.Time
	UpdatedAt time.Time
	DeletedAt gorm.DeletedAt `gorm:"index"`
}

func (User) TableName() string {
	return "l_users"
}

func main() {
	db, err := initDB()
	if err != nil {
		panic(err)
	}

	err = db.AutoMigrate(&User{})
	if err != nil {
		panic(err)
	}

	// 按需打开对应方法练习
	createUser(db)
	queryUser(db)
	updateUser(db)
}

func initDB() (*gorm.DB, error) {
	dsn := "root:mysql@(localhost:3306)/abc?charset=utf8mb4&parseTime=true&loc=Local"

	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		return nil, err
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, err
	}

	sqlDB.SetMaxOpenConns(20)
	sqlDB.SetMaxIdleConns(10)
	sqlDB.SetConnMaxLifetime(time.Hour)

	return db, nil
}

func createUser(db *gorm.DB) {
	user := User{
		Name:  "John Doe",
		Email: "john.doe@example.com",
	}

	result := db.Create(&user)
	if result.Error != nil {
		fmt.Println("Error creating user:", result.Error)
		return
	}

	fmt.Println("Create Successful, ID:", user.ID)
}

func queryUser(db *gorm.DB) {
	var user User

	err := db.Where("email = ?", "john.doe@example.com").First(&user).Error
	/*
	 * 这一段相当于：
	 * query := db.Where("email = ?", "john.doe@example.com")
	 * result := query.First(&user)
	 * err := result.Error
	 */
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			fmt.Println("用户不存在")
			return
		}

		fmt.Println("Error querying user:", err)
		return
	}

	fmt.Println("Query Successful")
	fmt.Println("ID:", user.ID)
	fmt.Println("Name:", user.Name)
	fmt.Println("Email:", user.Email)

	/*
	 * 这里 Age、Birthday 虽然都是指针：
	 * Age 有默认值 18，插入时没传也会由数据库填充，所以查询回来通常不是 nil。
	 * Birthday 没有默认值，插入时没传就是 NULL，所以查询回来是 nil。
	 */
	if user.Age != nil {
		fmt.Println("Age:", *user.Age)
	} else {
		fmt.Println("Age: nil")
	}

	if user.Birthday != nil {
		fmt.Println("Birthday:", user.Birthday.Format("2006-01-02"))
	} else {
		fmt.Println("Birthday: nil")
	}
}

func updateUser(db *gorm.DB) {
	// 由于 Updates 使用的是 map，不知道对应哪个模型，所以要通过 db.Model() 指定表。
	result := db.Model(&User{}).
		Where("email = ?", "john.doe@example.com").
		Updates(map[string]any{
			"name": "Jerry",
			"age":  22,
		})
	/*
	 * 传结构体，但要记住结构体默认跳过零值；map 会按你写的字段更新。
	 * 使用 Select，结构体形式可以更新零值。
	 * 如果没有 Select，GORM 会猜测你要更新的字段，零值会被跳过。
	 * 虽然模型里 Age 是 *int，但 map 更新写的是数据库列值，"age": 22 可以直接用 int。
	 */

	if result.Error != nil {
		fmt.Println("Error updating user:", result.Error)
		return
	}

	fmt.Println("Update Successful, RowsAffected:", result.RowsAffected)
}
```

## 总结

- `TableName()` 是方法，用于指定表名。
- `CreatedAt`、`UpdatedAt` 会自动维护。
- `DeletedAt` 会开启软删除。
- 指针字段用于区分 `nil` 和零值。
- map 更新会更新零值，结构体更新默认跳过零值。
- `UPDATE age = nil` 是更新为 `NULL`，不会触发默认值。
