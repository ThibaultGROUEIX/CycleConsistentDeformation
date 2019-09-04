from auxiliary.model_pointnetfeat import *
import torch


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2 * self.th(self.conv4(x))
        return x


class AE_AtlasNet(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024, nb_primitives=10, fixed_points=False):
        super(AE_AtlasNet, self).__init__()
        self.fixed_points = fixed_points
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.nb_primitives)])
        if self.fixed_points:
            rand_grid = Variable(torch.cuda.FloatTensor(1, 2, self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            self.rand_grid = rand_grid

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.fixed_points:
                rand_grid = Variable(
                    self.rand_grid.expand(x.size(0), 2, self.num_points // self.nb_primitives).contiguous().data)
            else:
                rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.nb_primitives))
                rand_grid.data.uniform_(0, 1)

            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()


# TEST with spheric noise
class AE_AtlasNet_SPHERE(nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024, nb_primitives=1, fixed_points=False):
        super(AE_AtlasNet_SPHERE, self).__init__()
        self.fixed_points = fixed_points

        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(num_points, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=3 + self.bottleneck_size) for i in range(0, self.nb_primitives)])
        if self.fixed_points:
            rand_grid = Variable(
                torch.cuda.FloatTensor(1, 3, self.num_points // self.nb_primitives))  # sample points randomly
            rand_grid.data.normal_(0, 1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
            self.rand_grid = rand_grid

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.fixed_points:
                rand_grid = Variable(
                    self.rand_grid.expand(x.size(0), 3, self.num_points // self.nb_primitives).contiguous().data)
            else:
                rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 3,
                                                            self.num_points // self.nb_primitives))  # sample points randomly
                rand_grid.data.normal_(0, 1)
                rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True)).expand(x.size(0), 3,
                                                                                                          self.num_points // self.nb_primitives)
                # assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x.size(0), grid.size(1), grid.size(2)).contiguous())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), grid.size(2)).contiguous()
            y = torch.cat((grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = grid.expand(x.size(0), grid.size(1), grid.size(2)).contiguous()
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), grid.size(2)).contiguous()
            y = torch.cat((grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2, 1).contiguous()
