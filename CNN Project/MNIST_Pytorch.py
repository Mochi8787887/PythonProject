# 參考網站: 
# 1. https://ithelp.ithome.com.tw/articles/10329534

# 前置操作: 安裝 PyTorch及其相關庫
pip install torch torchvision

# 引入函式庫
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm  # 導入tqdm函數庫
import matplotlib.pyplot as plt


# -----------------------------定義數據預處理操作-----------------------------
# 將圖像轉換為Tensor格式並進行歸一化處理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# -----------------------------下載訓練集和測試集-----------------------------
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# -----------------------------定義卷積神經網絡模型-----------------------------
# 定義了一個CNN模型，包括兩個卷積層和兩個全連接層
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()


# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# -----------------------------訓練模型----------------------------
train_losses = []  # 儲存每個epoch的訓練損失
for epoch in range(10):  # 訓練10個epoch
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):  # 使用tqdm來顯示進度條
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)
    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}')

print('Finished Training')


# -----------------------------測試模型性能----------------------------
correct = 0
total = 0
test_losses = []  # 儲存每個epoch的測試損失
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss = criterion(outputs, labels)
        test_losses.append(test_loss.item())

print(f'Accuracy on test images: {100 * correct / total}%')


# -----------------------------繪製Loss圖----------------------------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
