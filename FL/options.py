import argparse
from PIL import Image
def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=1000, help="rounds of training")

    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--atk_label', type=int, default=6, help="attacking label")
    parser.add_argument('--fak_label', type=int, default=4, help="fake label")
    parser.add_argument('--atk_client', type=int, default=0, help="attacking client")
    parser.add_argument('--fak_num', type=int, default=1000, help="")
    parser.add_argument('--frac', type=float, default=0.3, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--glr', type=float, default=0.0005, help="learning rate")

    # other arguments
    parser.add_argument("--attack", action='store_true')
    parser.add_argument("--initiative", action='store_true') 
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--noniid1', action='store_true')
    parser.add_argument('--noniid2', action='store_true')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--atk_store',type=str,default='atk_store')

    args = parser.parse_args()
    return args
