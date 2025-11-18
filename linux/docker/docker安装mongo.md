拉镜像

```shell
docker pull mongo
```

创数据卷

```shell
docker volume create mongo-data 
```



运行容器并挂载数据卷

```shell
docker run --name ocagent-db -d -p 27017:27017 -v mongo-data:/data/db mongo
```



设置生产环境

```shell
docker run --name ocagent-db \
  -d \
  -p 27017:27017 \
  -v mongo-data:/data/db \
  -e MONGO_INITDB_ROOT_USERNAME=yh666 \
  -e MONGO_INITDB_ROOT_PASSWORD=25721488 \
  mongo
    
```



mongo的web页面

```
docker run -d \
  --name ocagent-mongo-express \
  --link ocagent-db:mongo \
  -p 8083:8081 \
  -e ME_CONFIG_MONGODB_ADMINUSERNAME=yh666 \
  -e ME_CONFIG_MONGODB_ADMINPASSWORD=25721488 \
  -e ME_CONFIG_MONGODB_SERVER=mongo \
  mongo-express
    
```

`localhost:8083`输入`用户名:admin`、`密码:pass`即可。

