本地：

```shell
ssh-keygen -t rsa
```

将本地公钥存到服务器`.ssh/autoirized_keys`里面

```shell
echo "公钥" >> ~/.ssh/authorized_keys
```

