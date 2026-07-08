# GORM CRUD、查询分页、删除和批量操作

本文基于 `GORM框架.md` 中的 `User` 模型继续。

## 模型

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

func (User) TableName() string {
	return "ll_users"
}
```

`DeletedAt gorm.DeletedAt` 会开启软删除。

CRUD：

| 操作 | SQL | GORM |
| ---- | --- | ---- |
| Create | INSERT | `Create` |
| Read | SELECT | `First`、`Find` |
| Update | UPDATE | `Update`、`Updates` |
| Delete | DELETE | `Delete` |

GORM 不是不用 SQL，而是用 Go 方法帮你生成 SQL。

## Create

新增单条：

```go
func createUser(db *gorm.DB) {
	age := 20

	user := User{
		Name:  "Tom",
		Email: "tom@example.com",
		Age:   &age,
	}

	result := db.Create(&user)
	if result.Error != nil {
		fmt.Println("新增失败:", result.Error)
		return
	}

	fmt.Println("ID:", user.ID)
	fmt.Println("RowsAffected:", result.RowsAffected)
}
```

也可以只取错误：

```go
err := db.Create(&user).Error
```

`Create(&user)` 需要传指针，因为自增 ID 会回填到 `user.ID`。

大致 SQL：

```sql
INSERT INTO ll_users (name, email, age, created_at, updated_at)
VALUES ('Tom', 'tom@example.com', 20, NOW(), NOW());
```

### 零值和默认值

```go
Age *int `gorm:"column:age;type:int;default:18"`
```

```go
Age: nil
```

表示没传，让数据库默认值生效。

```go
age := 0
Age: &age
```

表示明确传入 `0`。

如果是普通 `int`：

```go
Age int `gorm:"default:18"`
```

创建时 `Age: 0` 是零值，GORM 可能让数据库默认值生效。

记法：

```text
int  + default + 0   -> 可能触发默认值
*int + default + nil -> 触发默认值
*int + default + &0  -> 保存 0
```

`Where` 中的零值不会被忽略，`Where("age = ?", 0)` 就是查询 `age = 0`。

### 自增 ID 跳号

如果插入失败，例如唯一索引冲突：

```text
Duplicate entry 'tom@example.com'
```

MySQL 可能已经申请了自增 ID。失败后自增值通常不会回退。

所以表里出现：

```text
id = 1
id = 3
```

是正常的。

```text
AUTO_INCREMENT 保证唯一递增，不保证连续。
```

## Read

### 查询单条

```go
func queryUserByEmail(db *gorm.DB, email string) {
	var user User

	result := db.Where("email = ?", email).First(&user)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			fmt.Println("用户不存在")
			return
		}

		fmt.Println("查询失败:", result.Error)
		return
	}

	fmt.Println(user.ID, user.Name, user.Email)
}
```

大致 SQL：

```sql
SELECT * FROM ll_users
WHERE email = 'tom@example.com'
  AND deleted_at IS NULL
ORDER BY id
LIMIT 1;
```

`First` 查不到时会返回 `gorm.ErrRecordNotFound`。

### 查询多条

```go
func queryUsers(db *gorm.DB, age int) {
	var users []User

	result := db.Where("age >= ?", age).Find(&users)
	if result.Error != nil {
		fmt.Println("查询失败:", result.Error)
		return
	}

	fmt.Println("数量:", len(users))
}
```

区别：

```text
First + User   -> 查一条
Find  + []User -> 查多条
```

`Find` 查不到时通常不会返回 `ErrRecordNotFound`，而是空切片。

### Where 零值

```go
email := ""
db.Where("email = ?", email).First(&user)
```

对应：

```sql
WHERE email = ''
```

不会自动忽略条件。

如果参数为空时不想加条件，需要自己判断：

```go
query := db.Model(&User{})

if email != "" {
	query = query.Where("email = ?", email)
}

result := query.First(&user)
```

如果 `email == ""`，最后会变成：

```go
db.Model(&User{}).First(&user)
```

也就是查表中第一条未软删除记录。

### Model

`Model(&User{})` 只指定表，不指定条件。

```go
db.Model(&User{}).First(&user)
```

大致是：

```sql
SELECT * FROM ll_users
WHERE deleted_at IS NULL
ORDER BY id
LIMIT 1;
```

查询时如果已经传入结果结构体，一般不用写 `Model`：

```go
var user User
db.First(&user)

var users []User
db.Find(&users)
```

更新、统计时通常需要：

```go
db.Model(&User{}).Update("age", 20)
db.Model(&User{}).Count(&total)
```

## Update

更新需要三部分：

```text
表：Model(&User{})
条件：Where(...)
字段：Update / Updates
```

单字段：

```go
func updateUserAge(db *gorm.DB, email string) {
	result := db.Model(&User{}).
		Where("email = ?", email).
		Update("age", 22)

	if result.Error != nil {
		fmt.Println("更新失败:", result.Error)
		return
	}

	if result.RowsAffected == 0 {
		fmt.Println("没有找到符合条件的用户")
		return
	}
}
```

多字段推荐 map：

```go
result := db.Model(&User{}).
	Where("email = ?", email).
	Updates(map[string]any{
		"name": "Jerry",
		"age":  25,
	})
```

`map` 更新会更新零值：

```go
Updates(map[string]any{
	"name": "",
	"age":  0,
})
```

结构体更新默认跳过零值：

```go
db.Model(&User{}).
	Where("email = ?", email).
	Updates(User{
		Name: "",
	})
```

`Name` 不会被更新。需要强制更新时：

```go
db.Model(&User{}).
	Where("email = ?", email).
	Select("Name").
	Updates(User{
		Name: "",
	})
```

总结：

```text
Updates(map)    -> 零值也更新
Updates(struct) -> 零值默认跳过
```

## 查询条件

多个 `Where` 默认用 `AND`：

```go
db.Where("age >= ?", minAge).
	Where("age <= ?", maxAge).
	Where("name LIKE ?", "%"+name+"%").
	Find(&users)
```

也可以写在一起：

```go
db.Where("age >= ? AND age <= ? AND name LIKE ?", minAge, maxAge, "%"+name+"%").
	Find(&users)
```

动态条件：

```go
func queryUsers(db *gorm.DB, minAge int, maxAge int, name string) {
	var users []User

	query := db.Model(&User{})

	if minAge > 0 {
		query = query.Where("age >= ?", minAge)
	}
	if maxAge > 0 {
		query = query.Where("age <= ?", maxAge)
	}
	if name != "" {
		query = query.Where("name LIKE ?", "%"+name+"%")
	}

	result := query.Find(&users)
	if result.Error != nil {
		fmt.Println("查询失败:", result.Error)
		return
	}

	fmt.Println("数量:", len(users))
}
```

`query = query.Where(...)` 表示在原查询上继续追加条件。

`LIKE`：

```go
db.Where("name LIKE ?", "%Tom%").Find(&users)
```

- `"Tom%"`：以 `Tom` 开头。
- `"%Tom"`：以 `Tom` 结尾。
- `"%Tom%"`：包含 `Tom`。

`IN`：

```go
db.Where("id IN ?", []uint{1, 2, 3}).Find(&users)
```

`OR`：

```go
db.Where("name = ?", "Tom").
	Or("name = ?", "Jerry").
	Find(&users)
```

排序：

```go
db.Order("id ASC").Find(&users)
db.Order("id DESC").Find(&users)
db.Order("age DESC").Order("id ASC").Find(&users)
```

前面大多是在拼 SQL：

```text
Model / Where / Order / Limit / Offset
```

真正执行 SQL：

```text
Create / First / Find / Update / Updates / Delete / Count / Scan
```

## Delete

模型有：

```go
DeletedAt gorm.DeletedAt `gorm:"index"`
```

所以默认删除是软删除。

```go
db.Delete(&User{}, 3)
```

大致 SQL：

```sql
UPDATE ll_users
SET deleted_at = 当前时间
WHERE id = 3
  AND deleted_at IS NULL;
```

按 ID 删除：

```go
func deleteUserByID(db *gorm.DB, id uint) {
	result := db.Delete(&User{}, id)
	if result.Error != nil {
		fmt.Println("删除失败:", result.Error)
		return
	}

	fmt.Println("删除数量:", result.RowsAffected)
}
```

按条件删除：

```go
result := db.Where("email = ?", email).Delete(&User{})
```

先查再删：

```go
var user User
db.Where("email = ?", email).First(&user)
db.Delete(&user)
```

查询包含软删除：

```go
db.Unscoped().Find(&users)
```

只查已删除：

```go
db.Unscoped().
	Where("deleted_at IS NOT NULL").
	Find(&users)
```

物理删除：

```go
db.Unscoped().Delete(&User{}, id)
```

对应：

```sql
DELETE FROM ll_users WHERE id = 3;
```

批量删除：

```go
func deleteUsersByIDs(db *gorm.DB, ids []uint) {
	if len(ids) == 0 {
		return
	}

	result := db.Where("id IN ?", ids).Delete(&User{})
	if result.Error != nil {
		fmt.Println("批量删除失败:", result.Error)
		return
	}

	fmt.Println("删除数量:", result.RowsAffected)
}
```

## 批量新增和批量更新

批量新增：

```go
func createUsers(db *gorm.DB) {
	age1 := 18
	age2 := 20

	users := []User{
		{Name: "Tom", Email: "tom@example.com", Age: &age1},
		{Name: "Jerry", Email: "jerry@example.com", Age: &age2},
	}

	result := db.Create(&users)
	if result.Error != nil {
		fmt.Println("批量新增失败:", result.Error)
		return
	}

	for _, user := range users {
		fmt.Println(user.ID, user.Name)
	}
}
```

分批新增：

```go
db.CreateInBatches(&users, 100)
```

批量更新成同一个值：

```go
func updateUsersAge(db *gorm.DB, ids []uint) {
	if len(ids) == 0 {
		return
	}

	result := db.Model(&User{}).
		Where("id IN ?", ids).
		Update("age", 18)

	if result.Error != nil {
		fmt.Println("批量更新失败:", result.Error)
		return
	}
}
```

批量更新多个字段：

```go
db.Model(&User{}).
	Where("id IN ?", ids).
	Updates(map[string]any{
		"name": "Updated",
		"age":  20,
	})
```

不同用户更新不同值，普通 `Updates` 不适合，因为它会把所有符合条件的行改成同一组值。可以循环更新，最好放事务里。

```go
type UpdateUserItem struct {
	ID   uint
	Name string
	Age  int
}

func batchUpdateUsers(db *gorm.DB, items []UpdateUserItem) error {
	return db.Transaction(func(tx *gorm.DB) error {
		for _, item := range items {
			result := tx.Model(&User{}).
				Where("id = ?", item.ID).
				Updates(map[string]any{
					"name": item.Name,
					"age":  item.Age,
				})

			if result.Error != nil {
				return result.Error
			}
			if result.RowsAffected == 0 {
				return fmt.Errorf("用户不存在，id=%d", item.ID)
			}
		}

		return nil
	})
}
```

事务里要用 `tx`，不要混用外面的 `db`。

## 分页

分页核心：

```go
Limit(size)
Offset(offset)
```

公式：

```go
offset := (page - 1) * size
```

例子：

```text
page = 1, size = 10 -> offset = 0
page = 2, size = 10 -> offset = 10
page = 3, size = 10 -> offset = 20
```

简单分页：

```go
func listUsers(db *gorm.DB, page int, size int) {
	var users []User

	if page <= 0 {
		page = 1
	}
	if size <= 0 {
		size = 10
	}

	offset := (page - 1) * size

	result := db.Order("id DESC").
		Limit(size).
		Offset(offset).
		Find(&users)

	if result.Error != nil {
		fmt.Println("查询失败:", result.Error)
		return
	}
}
```

列表接口通常要查总数：

```go
var total int64
query := db.Model(&User{})

result := query.Count(&total)
if result.Error != nil {
	return
}

result = query.Order("id DESC").
	Limit(size).
	Offset(offset).
	Find(&users)
```

这里基于同一个 `query` 执行了两次 SQL：

```text
Count -> SELECT count(*)
Find  -> SELECT 当前页数据
```

`query` 可以理解为带着查询条件的查询对象，不是已经执行完的结果。

带条件分页：

```go
func listUsersByCondition(db *gorm.DB, minAge int, maxAge int, name string, page int, size int) {
	var users []User
	var total int64

	if page <= 0 {
		page = 1
	}
	if size <= 0 {
		size = 10
	}

	query := db.Model(&User{})

	if minAge > 0 {
		query = query.Where("age >= ?", minAge)
	}
	if maxAge > 0 {
		query = query.Where("age <= ?", maxAge)
	}
	if name != "" {
		query = query.Where("name LIKE ?", "%"+name+"%")
	}

	result := query.Count(&total)
	if result.Error != nil {
		fmt.Println("统计失败:", result.Error)
		return
	}

	offset := (page - 1) * size

	result = query.Order("id DESC").
		Limit(size).
		Offset(offset).
		Find(&users)

	if result.Error != nil {
		fmt.Println("查询失败:", result.Error)
		return
	}

	fmt.Println("总数:", total)
	fmt.Println("当前页数量:", len(users))
}
```

## 事务

事务表示一组操作要么全部成功，要么全部失败。

```go
err := db.Transaction(func(tx *gorm.DB) error {
	result := tx.Model(&User{}).
		Where("id = ?", 1).
		Update("age", 18)
	if result.Error != nil {
		return result.Error
	}

	return nil
})
```

规则：

```text
return nil   -> 提交事务
return error -> 回滚事务
```

适合事务的场景：

- 一次操作修改多张表。
- 一次操作执行多条更新或删除。
- 中途失败时，前面的操作必须撤销。
- 批量操作要求全部成功或全部失败。

手动事务：

```go
tx := db.Begin()

result := tx.Model(&User{}).
	Where("id = ?", 1).
	Update("age", 18)
if result.Error != nil {
	tx.Rollback()
	return
}

tx.Commit()
```

一般优先用 `db.Transaction`，更不容易忘记回滚。

## 业务函数封装

新增：

```go
type CreateUserReq struct {
	Name  string
	Email string
	Age   *int
}

func CreateUser(db *gorm.DB, req CreateUserReq) (*User, error) {
	user := User{
		Name:  req.Name,
		Email: req.Email,
		Age:   req.Age,
	}

	result := db.Create(&user)
	if result.Error != nil {
		return nil, result.Error
	}

	return &user, nil
}
```

查询：

```go
func GetUserByID(db *gorm.DB, id uint) (*User, error) {
	var user User

	result := db.First(&user, id)
	if result.Error != nil {
		if errors.Is(result.Error, gorm.ErrRecordNotFound) {
			return nil, nil
		}

		return nil, result.Error
	}

	return &user, nil
}
```

局部更新：

```go
type UpdateUserReq struct {
	Name *string
	Age  *int
}

func UpdateUser(db *gorm.DB, id uint, req UpdateUserReq) error {
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

	result := db.Model(&User{}).
		Where("id = ?", id).
		Updates(updates)

	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}

	return nil
}
```

删除：

```go
func DeleteUser(db *gorm.DB, id uint) error {
	result := db.Delete(&User{}, id)
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return gorm.ErrRecordNotFound
	}

	return nil
}
```

列表：

```go
type ListUserReq struct {
	Name   string
	MinAge *int
	MaxAge *int
	Page   int
	Size   int
}

type ListUserResp struct {
	Items []User
	Total int64
	Page  int
	Size  int
}

func ListUsers(db *gorm.DB, req ListUserReq) (*ListUserResp, error) {
	var users []User
	var total int64

	if req.Page <= 0 {
		req.Page = 1
	}
	if req.Size <= 0 {
		req.Size = 10
	}

	query := db.Model(&User{})

	if req.Name != "" {
		query = query.Where("name LIKE ?", "%"+req.Name+"%")
	}
	if req.MinAge != nil {
		query = query.Where("age >= ?", *req.MinAge)
	}
	if req.MaxAge != nil {
		query = query.Where("age <= ?", *req.MaxAge)
	}

	result := query.Count(&total)
	if result.Error != nil {
		return nil, result.Error
	}

	offset := (req.Page - 1) * req.Size

	result = query.Order("id DESC").
		Limit(req.Size).
		Offset(offset).
		Find(&users)
	if result.Error != nil {
		return nil, result.Error
	}

	return &ListUserResp{
		Items: users,
		Total: total,
		Page:  req.Page,
		Size:  req.Size,
	}, nil
}
```

## 总结

- `Create(&user)`：新增，回填自增 ID。
- `First(&user)`：查一条，查不到返回 `gorm.ErrRecordNotFound`。
- `Find(&users)`：查多条，查不到一般是空切片。
- `Model(&User{})`：指定表。
- `Where(...)`：指定条件。
- `Update`：更新单字段。
- `Updates(map)`：更新多字段，零值也更新。
- `Updates(struct)`：默认跳过零值。
- `Delete`：有 `DeletedAt` 时默认软删除。
- `Unscoped().Delete`：物理删除。
- `Count + Limit + Offset + Find`：分页常见写法。
- `Transaction`：保证一组操作全部成功或全部失败。
