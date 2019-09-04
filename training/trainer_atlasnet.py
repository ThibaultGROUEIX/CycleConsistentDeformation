import torch
import torch.optim as optim
import time
import my_utils
import model_atlasnet
import extension.get_chamfer as get_chamfer
import os
import dataset_shapenet
from termcolor import colored
import useful_losses as loss
from abstract_trainer import AbstractTrainer

class Trainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.init_fix()
        self.git_repo_path = "https://github.com/ThibaultGROUEIX/CycleConsistentDeformation/commit/"
        self.init_save_dict(opt)


    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        if self.opt.atlasnet == "PATCH":
            network = model_atlasnet.AE_AtlasNet(num_points = 2500)
        if self.opt.atlasnet == "SPHERE":
            network = model_atlasnet.AE_AtlasNet_SPHERE(num_points = 2500)

        network.cuda()  # put network on GPU
        network.apply(my_utils.weights_init)  # initialization of the weight
        if self.opt.atlasnet == "PATCH":
            if os.path.exists(self.opt.atlasPatch_path):
                network.load_state_dict(torch.load(self.opt.atlasPatch_path))
                print(" Previous network weights loaded!")
        if self.opt.atlasnet == "SPHERE":
            if os.path.exists(self.opt.atlasSphere_path):
                network.load_state_dict(torch.load(self.opt.atlasSphere_path))
                print(" Previous network weights loaded!")
        self.network = network

    def build_optimizer(self):
        """
        Create optimizer
        """
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.opt.lrate)
        if self.opt.reload:
            self.optimizer.load_state_dict(torch.load(f'{self.opt.checkpointname}'))
            my_utils.yellow_print("Reloaded optimizer")

    def build_dataset_train(self):
        """
        Create training dataset
        """
        self.dataset_train = dataset_shapenet.ShapeNetSeg(mode=self.opt.mode,
                                                                               knn=self.opt.knn,
                                                                               num_neighbors=self.opt.num_neighbors,
                                                                               normalization=self.opt.normalization,
                                                                               class_choice=self.opt.cat,
                                                                               data_augmentation_Z_rotation=True,
                                                                               data_augmentation_Z_rotation_range=40,
                                                                               anisotropic_scaling=self.opt.anisotropic_scaling,
                                                                               npoints=self.opt.number_points,
                                                                               random_translation=True,
                                                                               get_single_shape=True)
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.opt.batch_size,
                                                            shuffle=True, num_workers=int(self.opt.workers),
                                                            drop_last=True)
        self.len_dataset = len(self.dataset_train)


    def build_dataset_test(self):
        """
        Create testing dataset
        """
        self.dataset_test = dataset_shapenet.ShapeNetSeg(mode="TEST",
                                                                              normalization=self.opt.normalization,
                                                                              class_choice=self.opt.cat,
                                                                              data_augmentation_Z_rotation=False,
                                                                              data_augmentation_Z_rotation_range=40,
                                                                              npoints=self.opt.number_points,
                                                                              random_translation=False,
                                                                              get_single_shape=True)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.opt.batch_size,
                                                           shuffle=False, num_workers=int(self.opt.workers),
                                                           drop_last=True)
        self.len_dataset_test = len(self.dataset_test)


    def build_losses(self):
        """
        Create losses
        """
        self.distChamfer = get_chamfer.get(self.opt)
        self.forward_chamfer_atlasnet = loss.forward_chamfer_atlasnet

    def init_fix(self):
        """
        Create a useful vector for indexing batched pointcloud.
        """
        self.fix = torch.arange(0, self.opt.batch_size).view(self.opt.batch_size, 1).repeat(1,
                                                                                            self.opt.number_points).view(
            -1).long().cuda() * self.opt.number_points



    def train_iteration(self):
        self.optimizer.zero_grad()
        label1 = self.P1[:,:,6].contiguous()
        label1[0] = label1[0] - torch.min(label1[0]) + 1

        self.P1 = self.P1[:,:,:3].contiguous().cuda().float()
        P2, dist1_P2, dist2_P2, idx1_P2, idx2_P2 = self.forward_chamfer_atlasnet(self.network, self.P1, self.fix, distChamfer=self.distChamfer)
        loss_train_Deformation_ChamferL2 = loss.chamferL2(dist1_P2, dist2_P2)

        loss_train_total = loss_train_Deformation_ChamferL2

        loss_train_total.backward()
        self.log.update("loss_train_Deformation_ChamferL2", loss_train_Deformation_ChamferL2)
        self.log.update("loss_train_total", loss_train_total)
        self.optimizer.step()  # gradient update

        # VIZUALIZE
        if self.iteration % 50 == 1 and self.opt.display:
            self.visualizer.show_pointclouds(points=P2[0],  title="train_A_reconstructed")
            self.visualizer.show_pointclouds(points=self.P1[0], Y=label1[0], title="train_A")

        self.print_iteration_stats(loss_train_total)


    def train_epoch(self):
        self.log.reset()
        self.network.train()
        self.learning_rate_scheduler()
        start = time.time()
        iterator = self.dataloader_train.__iter__()
        self.reset_iteration()
        while True:
            try:
                P1, _, _, _ = iterator.next()
                self.P1 = P1
                self.increment_iteration()
            except:
                print(colored("end of train dataset", 'red'))
                break
            self.train_iteration()
        print("Ellapsed time : ", time.time() - start)

    def test_iteration(self):
        label1 = self.P1[:,:,6].contiguous()
        label1[0] = label1[0] - torch.min(label1[0]) + 1

        self.P1 = self.P1[:,:,:3].contiguous().cuda().float()

        P2, dist1_P2, dist2_P2, idx1_P2, idx2_P2 = self.forward_chamfer_atlasnet(self.network, self.P1, self.fix, distChamfer=self.distChamfer)
        loss_val_Deformation_ChamferL2 = loss.chamferL2(dist1_P2, dist2_P2)
        self.log.update("loss_val_Deformation_ChamferL2", loss_val_Deformation_ChamferL2)

        print(
            '\r' + colored('[%d: %d/%d]' % (self.epoch, self.iteration, self.len_dataset_test / (self.opt.batch_size)),
                           'red') +
            colored('loss_val_Deformation_ChamferL2:  %f' % loss_val_Deformation_ChamferL2.item(), 'yellow'),
            end='')

        if self.iteration % 60 == 1 and self.opt.display:
            self.visualizer.show_pointclouds(points=P2[0],title="val_A_reconstructed")
            self.visualizer.show_pointclouds(points=self.P1[0], Y=label1[0], title="val_A")


    def test_epoch(self):
        self.network.eval()
        iterator = self.dataloader_test.__iter__()
        self.reset_iteration()
        while True:
            self.increment_iteration()
            try:
                P1, _, _, _ = iterator.next()
                self.P1 = P1
            except:
                print(colored("end of val dataset", 'red'))
                break
            self.test_iteration()

        self.log.end_epoch()
        if self.opt.display:
            self.log.update_curves(self.visualizer.vis, self.opt.save_path)

