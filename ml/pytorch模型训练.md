将整个过程分成两部分：**Model**和**Solver**。**Model**作为神经网络模型的实现；**Solver**里面包含**Model**，作为封装Model的整个训练过程，包括训练、验证、模型参数保存等等。

同时，也可以将一些超参数专门写到**配置文件**里面，例如yaml格式的，这样也可以更好、更快的修正或验证！

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from logger import logger as log

### 神经网络模型
class MLP(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_hidden_layers: int,
            dropout_prob: float,
    ) -> None:
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        # 输入层到第一个隐藏层
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))

        # 后续隐藏层
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        
        x = self.fc_out(x)

        return x

### Solver
class Solver:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            device: torch.device,
            num_epochs: int,
    ) -> None:
        self.model = model
        self.train_loader =train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

        # 将模型移动到设备
        self.model.to(self.device)

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        for X_train, y_train in self.train_loader:
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self) -> float:
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_val, y_val in self.val_loader:
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                outputs = self.model(X_val)
                loss = self.criterion(outputs, y_val)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)
    
    def fit(self) -> None:
        log.info(f"Starting training on {self.device} for {self.num_epochs}")
        for epoch in range(self.num_epochs):
            avg_train_loss = self.train_epoch()
            avg_val_loss = self.validate_epoch()

            if (epoch + 1) % 100 == 0:
                log.info(f"Epoch[{epoch + 1}/{self.num_epochs}], Train loss: {avg_train_loss}, Val loss: {avg_val_loss}")
        log.info(f"Training finished.")
    
    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    X_batch = batch[0]
                else:
                    X_batch = batch
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_predictions.append(outputs.cpu())
        return torch.cat(all_predictions, dim=0)
    
    def save(self, path: str = "models/model.pth") -> None:
         torch.save(self.model.state_dict(), path)
         log.info(f"Model state_dict to {path}")

    def load(self, path: str = "models/model.pth") -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        log.info(f"Model state_dict loaded from {path}")

if __name__ == "__main__":
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    ### 数据
    a,b = 1, 2
    x1 = torch.randn((num_epochs, 1))
    x2 = torch.randn((num_epochs, 1))
    y = x1 * a + x2 * b + torch.normal(0, 0.2, (num_epochs, 1))
    x = torch.cat([x1, x2], dim=1)

    log.info(f"x: {x.shape}, y: {y.shape}")

    train_data = TensorDataset(x, y)
    train_set, val_set = random_split(train_data, [800, 200])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    log.info(f"Train set size: {len(train_loader)}, Val set size: {len(val_loader)}")
    train_set_batch, train_labels_batch = next(iter(train_loader))
    log.info(f"Train batch shape: {train_set_batch.shape}, Train labels batch size: {train_labels_batch.shape}")

    ### 模型初始化
    input_size = 2
    hidden_size = 64
    output_size = 1
    num_hidden_layers = 3
    dropout_prob = 0.0

    model = MLP(input_size,hidden_size,output_size,num_hidden_layers,dropout_prob)

    ### 优化器和损失器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ### 创建 Solver
    num_epochs = 1000
    solver = Solver(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    # solver.fit()
    # solver.save()
    solver.load()
    test_data_x = torch.tensor([[3,2],
                                [4,5]], dtype=torch.float32)
    log.info(f"{test_data_x.shape}")
    test_dataset = TensorDataset(test_data_x)
    test_loader = DataLoader(test_dataset,32)
    predis = solver.predict(test_loader)
    log.info(predis)
```



还实现了一个logger脚本，进行观察：

```python
# pip install loguru
from loguru import logger
import os

LOG_FORMAT = (

    "<cyan>{file.name}</cyan>:<cyan>{function}</cyan>:<b><magenta>{line}</magenta></b> - "
    "<level>{message}</level>"
)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "app.log")

logger.remove()
logger.add(
    log_path,
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
    enqueue=True,
    format=LOG_FORMAT,
    level="INFO"
)
logger.add(
    sink=lambda msg: print(msg, end=""),
    format=LOG_FORMAT,
    level="INFO",
    colorize=True
)
```

