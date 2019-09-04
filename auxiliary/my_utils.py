import random
import numpy as np
from termcolor import colored
import torch


def uniformize_sizes(knn):
    """
    input : knn :
    -- list of tensors of size(npoints x 3)
    N_point can be different for each sample
    It is used to compute the KNN in the dataloader.
    Since all pointcloud have a point (100, 100, 100) it doesn't change their chamfer distance and doesn't affect the nearest neighbors.

    :param knn: list of tensors of size(npoints x 3)
    :return: Return a tensor of size B, max_npoints, 3. The missing values are filled with 100.
    """

    sizes = [data.size(0) for data in knn]
    scale = [data.max() for data in knn]
    scale = max(scale)
    max_sizes = max(sizes) + 1
    knn_new_list = []
    for data in knn:
        data = data.squeeze()
        new_tensor = torch.zeros((max_sizes, 3)) + 100
        new_tensor[:data.size(0)] = data[:, :3]
        knn_new_list.append(new_tensor.unsqueeze(0).float())

    return torch.cat(knn_new_list, 0)


class Min_k(object):
    def __init__(self, k_max):
        self.k_max = k_max

    def __call__(self, my_list):
        min_k_idx = np.argsort(np.array(my_list))[:self.k_max]
        min_k_values = [my_list[i] for i in min_k_idx]
        return min_k_idx, min_k_values


class Max_k(object):
    def __init__(self, k_max):
        self.k_max = k_max

    def __call__(self, my_list):
        max_k_idx = np.argsort(np.array(my_list))[-self.k_max:]
        max_k_values = [my_list[i] for i in max_k_idx]
        return list(reversed(max_k_idx)), list(reversed(max_k_values))


def convert_label_to_one_hot_torch(labels, num_categories):
    labels = labels.view(labels.nelement(), 1).long()
    labels_onehot = torch.FloatTensor(len(labels), num_categories).cuda()
    labels_onehot.zero_()
    labels_onehot.scatter_(1, labels, 1)
    return labels_onehot


def get_emsemble(pred_list):
    # input : torch array pred_list
    # output : numpy array of prediction
    n_ensemble, n_points = pred_list.size()
    n_cats = int(pred_list.max().item()) + 1
    one_hot = convert_label_to_one_hot_torch(pred_list.view(-1), n_cats)
    one_hot = one_hot.view(n_ensemble, n_points, n_cats).contiguous()
    one_hot[0] = one_hot[0] * 1.2  # Make sure the first prediction is favored in case of ties
    one_hot = one_hot.sum(dim=0).contiguous()
    preds = one_hot.argmax(1)
    return preds.cpu().numpy()


def int_2_boolean(x):
    if x == 1:
        return True
    else:
        return False


def plant_seeds(randomized_seed=False):
    if randomized_seed:
        manualSeed = random.randint(1, 10000)
    else:
        manualSeed = 1
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)


# initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def adjust_learning_rate(optimizer, epoch, phase):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch % phase == (phase - 1):
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] / 10.0


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grey_print(x):
    print(colored(x, "grey"))


def red_print(x):
    print(colored(x, "red"))


def green_print(x):
    print(colored(x, "green"))


def yellow_print(x):
    print(colored(x, "yellow"))


def blue_print(x):
    print(colored(x, "blue"))


def magenta_print(x):
    print(colored(x, "magenta"))


def cyan_print(x):
    print(colored(x, "cyan"))


def white_print(x):
    print(colored(x, "white"))


def print_arg(opt):
    cyan_print("PARAMETER: ")
    for a in opt.__dict__:
        print(
            "         "
            + colored(a, "yellow")
            + " : "
            + colored(str(opt.__dict__[a]), "cyan")
        )


if __name__ == "__main__":
    # To make your color choice reproducible, uncomment the following line:
    # random.seed(10)

    colors = get_colors(10)
    print("Your colors:", colors)
