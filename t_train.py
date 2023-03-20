import torch

from data import zidian_y, loader, zidian_xr, zidian_yr
from mask import mask_pad, mask_tril
from model import Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = Transformer()
model.cuda(device)
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)

def train():
    for epoch in range(1):
        for i, (x, y) in enumerate(loader):
            # x = [8, 50]
            # y = [8, 51]
            x = x.cuda(device)
            y = y.cuda(device)

            # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
            # [8, 50, 39]
            pred = model(x, y[:, :-1])

            # [8, 50, 39] -> [400, 39]
            pred = pred.reshape(-1, 39)

            # [8, 51] -> [400]
            y = y[:, 1:].reshape(-1)

            # 忽略pad
            select = y != zidian_y['<PAD>']
            pred = pred[select]
            y = y[select]

            loss = loss_func(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 200 == 0:
                # [select, 39] -> [select]
                pred = pred.argmax(1)
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optim.param_groups[0]['lr']
                print(epoch, i, lr, loss.item(), accuracy)

        sched.step()

train()

# 保存模型
torch.save(model, 't_model.pt')