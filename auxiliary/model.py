from auxiliary.model_pointnetfeat import *
import collections
import my_utils


class MetaVector(nn.Module):
    def __init__(self, bottleneck_size=1024, target_size=1024):
        self.bottleneck_size = bottleneck_size
        self.target_size = target_size
        super(MetaVector, self).__init__()
        self.fc1 = nn.Linear(self.bottleneck_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.target_size)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class FC_Layer(nn.Module):
    def __init__(self, source_size=1024, target_size=1024, activation="ReLU", batchnorm=True):
        self.source_size = source_size
        self.target_size = target_size
        self.activation = activation
        self.batchnorm = batchnorm
        super(FC_Layer, self).__init__()
        self.conv1 = torch.nn.Conv1d(source_size, target_size, 1)
        self.bn1 = torch.nn.BatchNorm1d(target_size)
        if self.activation == "ReLU":
            self.activation_function = nn.ReLU()
        elif self.activation == "Tanh":
            self.activation_function = nn.Tanh()
        else:
            print("unrecognized activation function")

    def forward(self, x):
        if self.batchnorm:
            return self.activation_function(self.bn1(self.conv1(x)))
        else:
            return self.activation_function(self.conv1(x))


class Meta_FC_Layer(nn.Module):
    """
    This corresponds to Arxiv v1 of the paper. x = x * weight + bias
    """

    def __init__(self, source_size=1024, target_size=1024, activation="ReLU", batchnorm=True, cursor_Wstart=0,
                 cursor_Wend=1, cursor_Bstart=0, cursor_Bend=1):
        super(Meta_FC_Layer, self).__init__()

        self.fc = FC_Layer(source_size, target_size, activation, batchnorm)
        self.target_size = target_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.cursor_Wstart = cursor_Wstart
        self.cursor_Wend = cursor_Wend
        self.cursor_Bstart = cursor_Bstart
        self.cursor_Bend = cursor_Bend

    def forward(self, args):
        x, parameters = args
        weight = parameters[:, self.cursor_Wstart:self.cursor_Wend].contiguous()
        bias = parameters[:, self.cursor_Bstart:self.cursor_Bend].contiguous()

        bias = bias.unsqueeze(2).expand(x.size(0), bias.size(1), x.size(2)).contiguous()
        weight = weight.unsqueeze(2).expand(x.size(0), weight.size(1), x.size(2)).contiguous()

        x = x * weight + bias
        return [self.fc(x), parameters]


class Meta_FC_Layer_2(nn.Module):
    """
    This corresponds to Arxiv v2 of the paper. x = x bmm weight + bias. It implements a predicted FC.
    """

    def __init__(self, source_size=1024, target_size=1024, activation="ReLU", batchnorm=True, cursor_Wstart=0,
                 cursor_Wend=1, cursor_Bstart=0, cursor_Bend=1):
        super(Meta_FC_Layer_2, self).__init__()

        self.fc = FC_Layer(source_size, target_size, activation, batchnorm)
        self.target_size = target_size
        self.activation = activation
        self.batchnorm = batchnorm
        self.cursor_Wstart = cursor_Wstart
        self.cursor_Wend = cursor_Wend
        self.cursor_Bstart = cursor_Bstart
        self.cursor_Bend = cursor_Bend
        self.bn0 = torch.nn.BatchNorm1d(source_size)

    def forward(self, args):
        x, parameters = args
        weight = parameters[:, self.cursor_Wstart:self.cursor_Wend].contiguous()
        bias = parameters[:, self.cursor_Bstart:self.cursor_Bend].contiguous()

        bias = bias.unsqueeze(2).expand(x.size(0), bias.size(1), x.size(2)).contiguous()
        weight = weight.view(x.size(0), bias.size(1), bias.size(1)).contiguous() + torch.eye(bias.size(1)).unsqueeze(
            0).repeat(x.size(0), 1, 1).cuda()

        x = x + bias
        x = x.transpose(2, 1).contiguous()
        x = torch.bmm(x, weight)
        x = x.transpose(2, 1).contiguous()
        x = F.relu(self.bn0(x))  # + bias

        return [self.fc(x), parameters]


class MetaPointGenCon(nn.Module):
    """
    This corresponds to Arxiv v1 of the paper. x = x * weight + bias
    """

    def __init__(self, bottleneck_size=2500, hidden_sizes=[256, 256, 256], resnet_layers=True):
        self.bottleneck_size = bottleneck_size
        super(MetaPointGenCon, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.resnet_layers = resnet_layers
        self.num_parameters = 3 + 3
        for i in hidden_sizes:
            self.num_parameters = self.num_parameters + i + i

        self.parameter_predictor = MetaVector(bottleneck_size, self.num_parameters)

        hidden_layer = collections.OrderedDict()
        hidden_layer['fc_0'] = Meta_FC_Layer(3, hidden_sizes[0], cursor_Wstart=0, cursor_Wend=3, cursor_Bstart=3,
                                             cursor_Bend=6)
        cursor = 6

        for i, size in enumerate(hidden_sizes[:-1]):
            hidden_layer['fc_' + str(i + 1)] = Meta_FC_Layer(hidden_sizes[i], hidden_sizes[i + 1], cursor_Wstart=cursor,
                                                             cursor_Wend=cursor + size, cursor_Bstart=cursor + size,
                                                             cursor_Bend=cursor + 2 * size)
            cursor = cursor + 2 * size

        hidden_layer['fc_end'] = Meta_FC_Layer(hidden_sizes[-1], 3, activation="Tanh", batchnorm=False,
                                               cursor_Wstart=cursor,
                                               cursor_Wend=cursor + hidden_sizes[-1],
                                               cursor_Bstart=cursor + hidden_sizes[-1],
                                               cursor_Bend=cursor + 2 * hidden_sizes[-1])

        self.hidden_layer = hidden_layer
        self.MLP = nn.Sequential(self.hidden_layer)

    def forward(self, x, feature_vector):
        parameters = self.parameter_predictor(feature_vector)
        x, _ = self.MLP([x, parameters])
        return 2 * x


class MetaPointGenCon2(nn.Module):
    """
    This corresponds to Arxiv v2 of the paper. x = x bmm weight + bias. It implements a predicted FC.
    """

    def __init__(self, bottleneck_size=2500, hidden_sizes=[256, 256, 256], resnet_layers=True):
        self.bottleneck_size = bottleneck_size
        super(MetaPointGenCon2, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.resnet_layers = resnet_layers
        self.num_parameters = 9 + 3
        for i in hidden_sizes:
            self.num_parameters = self.num_parameters + i * i + i

        self.parameter_predictor = MetaVector(bottleneck_size, self.num_parameters)

        hidden_layer = collections.OrderedDict()
        hidden_layer['fc_0'] = Meta_FC_Layer_2(3, hidden_sizes[0], cursor_Wstart=0, cursor_Wend=9, cursor_Bstart=9,
                                               cursor_Bend=12)
        cursor = 12

        for i, size in enumerate(hidden_sizes[:-1]):
            hidden_layer['fc_' + str(i + 1)] = Meta_FC_Layer_2(hidden_sizes[i], hidden_sizes[i + 1],
                                                               cursor_Wstart=cursor,
                                                               cursor_Wend=cursor + size * size,
                                                               cursor_Bstart=cursor + size * size,
                                                               cursor_Bend=cursor + size + size * size)
            cursor = cursor + size * size + size

        hidden_layer['fc_end'] = Meta_FC_Layer_2(hidden_sizes[-1], 3, activation="Tanh", batchnorm=False,
                                                 cursor_Wstart=cursor,
                                                 cursor_Wend=cursor + hidden_sizes[-1] * hidden_sizes[-1],
                                                 cursor_Bstart=cursor + hidden_sizes[-1] * hidden_sizes[-1],
                                                 cursor_Bend=cursor + hidden_sizes[-1] * hidden_sizes[-1] +
                                                             hidden_sizes[-1])

        self.hidden_layer = hidden_layer
        self.MLP = nn.Sequential(self.hidden_layer)

    def forward(self, x, feature_vector):
        parameters = self.parameter_predictor(feature_vector)
        x, _ = self.MLP([x, parameters])
        return 2 * x


class transpose(nn.Module):
    def __init__(self):
        super(transpose, self).__init__()

    def forward(self, x):
        return x.transpose(1, 2).contiguous()


class AE_Meta_AtlasNet(nn.Module):
    def __init__(self, num_points=6890, encoder_type="Pointnet", bottleneck_size=1024, nb_primitives=1,
                 hidden_sizes=[64, 64, 64, 64, 64], resnet_layers=True, skip_connections=False):
        self.hidden_sizes = hidden_sizes
        self.resnet_layers = resnet_layers
        super(AE_Meta_AtlasNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder_type = encoder_type
        print("Using encoder type : " + self.encoder_type)

        self.point_encoder_1 = PointNetfeat(num_points, global_feat=True, trans=False)
        self.point_encoder_2 = PointNetfeat(num_points, global_feat=True, trans=False)
        self.encoder = nn.Sequential(self.point_encoder_1,
                                     nn.Linear(1024, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU()
                                     )
        self.encoder2 = nn.Sequential(self.point_encoder_2,
                                      nn.Linear(1024, 512),
                                      nn.BatchNorm1d(512),
                                      nn.ReLU()
                                      )

        self.decoder = MetaPointGenCon2(bottleneck_size=self.bottleneck_size, hidden_sizes=hidden_sizes,
                                        resnet_layers=resnet_layers)
        self.skip_connections = skip_connections
        if self.skip_connections:
            my_utils.yellow_print("Enable Skip_connections in pointcloud MLP")
            self.forward = self.forward_resnet
        else:
            my_utils.yellow_print("Desable Skip_connections in pointcloud MLP")
            self.forward = self.forward_classic

    def forward_classic(self, x, y):
        x_latent = self.encoder(x)
        y_latent = self.encoder2(y)
        latent = torch.cat([x_latent, y_latent], 1)
        y_recon = self.decoder(x, latent)
        return y_recon.contiguous()

    def forward_classic_with_latent(self, x, y, x_latent=0, y_latent=0):
        if x_latent is 0:
            x_latent = self.encoder(x)
        if y_latent is 0:
            y_latent = self.encoder2(y)
        latent = torch.cat([x_latent, y_latent], 1)
        y_recon = self.decoder(x, latent)
        return y_recon.contiguous(), x_latent, y_latent

    def forward_resnet(self, x, y):
        return x + self.forward_classic(x, y)

    def decode(self, x, latent=None, x_latent=None, y_latent=None):
        if latent is None:
            latent = torch.cat([x_latent, y_latent], 1)
        y_recon = self.decoder(x, latent)
        return y_recon.contiguous().transpose(2, 1).contiguous()

    def encode(self, x, y):
        x_latent = self.encoder(x)
        y_latent = self.encoder2(y)
        latent = torch.cat([x_latent, y_latent], 1)
        return latent


if __name__ == '__main__':
    model = AE_Meta_AtlasNet(encoder_type="Pointnet").cuda()
    sim_data = Variable(torch.rand(16, 3, 2048)).cuda()
    sim_data_2 = Variable(torch.rand(16, 3, 2048)).cuda()
    model(sim_data, sim_data_2)
