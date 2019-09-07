#--*-- encoding: UTF-8 --*--
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time
import sys

#定义神经网络
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output

#当命令行参数中包含train时才执行训练
if len(sys.argv)>1 and sys.argv[1] == 'train':
    print("Start training ...")
    #设置随机数种子，保证每次结果一致
    torch.manual_seed(1)

    EPOCH = 1           #批数量，学习几轮
    BATCH_SIZE = 50     #批大小，一次取50个数据
    LR = 0.001          #学习率
    DOWNLOAD_MNIST = True

    #训练数据和测试数据，如果没有数据则自动下载
    train_data = torchvision.datasets.MNIST(
        root = './mnist/',
        train = True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
    )
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

    #训练数据生成器
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255.0  #将图片的像素转换至0-1区间
    test_y = test_data.targets #图片上的数字


    cnn = CNN()

    optimzer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    time_start = time.time()
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            arr = torch.max(output, dim=1)[1].data.numpy()


            if step % 50 == 0:
                test_output = cnn(test_x)
                predict_y = torch.max(test_output, 1)[1].data.numpy()

                accuracy = float((predict_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
                cost = time.time() - time_start

                print("Epoch: %d | train loss: %.4f | test accuracy: %.2f | time cost: %.6f"%(epoch, loss, accuracy, cost))


    torch.save(cnn.state_dict(), 'net_params.pkl')
else:
    print('my_mnist.py: no command found')
