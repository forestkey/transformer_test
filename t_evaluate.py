import torch

from data import zidian_y, loader, zidian_xr, zidian_yr
from mask import mask_pad, mask_tril

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 预测函数
def predict(model, x):
    # x = [1, 50]
    model.eval()
    
    x = x.cuda(0)
    # [1, 1, 50, 50]
    mask_pad_x = mask_pad(x)

    # 初始化输出,这个是固定值
    # [1, 50]
    # [[0,2,2,2...]]
    target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0)
    target = target.cuda(0)

    # x编码,添加位置信息
    # [1, 50] -> [1, 50, 32]
    x = model.embed_x(x)

    # 编码层计算,维度不变
    # [1, 50, 32] -> [1, 50, 32]
    x = model.encoder(x, mask_pad_x)

    # 遍历生成第1个词到第49个词
    for i in range(49):
        # [1, 50]
        y = target

        # [1, 1, 50, 50]
        mask_tril_y = mask_tril(y)

        # y编码,添加位置信息
        # [1, 50] -> [1, 50, 32]
        y = model.embed_y(y)

        # 解码层计算,维度不变
        # [1, 50, 32],[1, 50, 32] -> [1, 50, 32]
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,39分类
        # [1, 50, 32] -> [1, 50, 39]
        out = model.fc_out(y)

        # 取出当前词的输出
        # [1, 50, 39] -> [1, 39]
        out = out[:, i, :]

        # 取出分类结果
        # [1, 39] -> [1]
        out = out.argmax(dim=1).detach()

        # 以当前词预测下一个词,填到结果中
        target[:, i + 1] = out

    return target

# 测试
for i, (x, y) in enumerate(loader):
    break

t_model = torch.load('t_model.pt')
t_model.cuda(0)

for i in range(8):
    print(i)
    print(''.join([zidian_xr[i] for i in x[i].tolist()]))
    print(''.join([zidian_yr[i] for i in y[i].tolist()]))
    print(''.join([zidian_yr[i] for i in predict(t_model, x[i].unsqueeze(0))[0].tolist()]))