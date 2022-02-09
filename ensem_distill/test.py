import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

#测试集 准确率
def test_img(net, datatest, args):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target).item()
            # get the index of the max log-probability
            y_pred = torch.max(log_probs, 1)[1].data.squeeze()
            correct += (y_pred == target).sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

    return accuracy, test_loss