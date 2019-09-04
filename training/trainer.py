import torch
import torch.optim as optim
import time
import my_utils
import model
import extension.get_chamfer as get_chamfer
import dataset_shapenet
from termcolor import colored
import miou_shape
import triplet_point_cloud
import couple_point_cloud
import useful_losses as loss
from abstract_trainer import AbstractTrainer

class Trainer(AbstractTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        self.init_fix()
        self.print_loss_info()
        self.git_repo_path = "https://github.com/ThibaultGROUEIX/CycleConsistentDeformation/commit/"
        self.init_save_dict(opt)


    def print_loss_info(self):
        my_utils.cyan_print("LOSS")
        my_utils.cyan_print(" lambda_chamfer : " + str(self.opt.lambda_chamfer))
        my_utils.cyan_print(" chamfer_loss_type : " + str(self.opt.chamfer_loss_type))
        my_utils.cyan_print(" lambda_cycle_2 : " + str(self.opt.lambda_cycle_2))
        my_utils.cyan_print(" lambda_cycle_3 : " + str(self.opt.lambda_cycle_3))
        my_utils.cyan_print(" lambda_reconstruct : " + str(self.opt.lambda_reconstruct))

    def build_network(self):
        """
        Create network architecture. Refer to auxiliary.model
        :return:
        """
        network = model.AE_Meta_AtlasNet(hidden_sizes=self.opt.hidden_sizes,
                                                skip_connections=self.opt.skip_connections,
                                                encoder_type=self.opt.encoder_type,
                                                resnet_layers=self.opt.resnet_layers)
        network.cuda()  # put network on GPU
        network.apply(my_utils.weights_init)  # initialization of the weight
        if self.opt.model != "":
            try:
                network.load_state_dict(torch.load(self.opt.model))
                print(" Previous network weights loaded! From ", self.opt.model)
            except:
                print("Failed to reload " , self.opt.model)
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
                                                                               random_translation=True)
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
                                                                              random_translation=False)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.opt.batch_size,
                                                           shuffle=False, num_workers=int(self.opt.workers),
                                                           drop_last=True)
        self.len_dataset_test = len(self.dataset_test)

    def build_dataset_train_for_matching(self):
        """
        Create training dataset for matching used at inference
        """
        self.dataset_train = dataset_shapenet.ShapeNetSeg(mode="TRAIN", knn=False,
                                                                               normalization=self.opt.normalization,
                                                                               class_choice=self.opt.cat,
                                                                               npoints=self.opt.number_points_eval,
                                                                               data_augmentation_Z_rotation=False,
                                                                               anisotropic_scaling=False,
                                                                               sample=False,
                                                                               shuffle=self.opt.randomize,
                                                                               random_translation=False,
                                                                               get_single_shape=True)
        self.dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=1,
                                                            shuffle=self.opt.randomize,
                                                            num_workers=int(self.opt.workers), drop_last=False)
        self.len_dataset = len(self.dataset_train)

    def build_dataset_test_for_matching(self):
        """
        Create testing dataset for matching used at inference
        """
        self.dataset_test = dataset_shapenet.ShapeNetSeg(mode="TEST", knn=False,
                                                                              normalization=self.opt.normalization,
                                                                              class_choice=self.opt.cat,
                                                                              npoints=self.opt.number_points_eval,
                                                                              data_augmentation_Z_rotation=False,
                                                                              anisotropic_scaling=False,
                                                                              sample=False,
                                                                              random_translation=False,
                                                                              get_single_shape=True)
        self.dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=1,
                                                           shuffle=False, num_workers=1, drop_last=False)
        self.len_dataset_test = len(self.dataset_test)
        self.parts = self.dataset_train.part_category[self.opt.cat]

    def build_losses(self):
        """
        Create losses
        """
        self.distChamfer = get_chamfer.get(self.opt)
        self.forward_chamfer_bypart = loss.ForwardChamferByPart(self.opt.chamfer_loss_type)
        self.forward_chamfer = loss.forward_chamfer

    def init_fix(self):
        """
        Create a useful vector for indexing batched pointcloud.
        """
        self.fix = torch.arange(0, self.opt.batch_size).view(self.opt.batch_size, 1).repeat(1,
                                                                                            self.opt.number_points).view(
            -1).long().cuda() * self.opt.number_points



    def triplet_chamfer_loss(self):
        """
        Compute all reconstruction on a triplet P1,P2,P3 without part supervision
        """
        self.triplet.add_P2P1(
            *self.forward_chamfer(self.network, self.triplet.P1, self.triplet.P2, self.fix, True, 0, 0,
                                  self.distChamfer))
        self.triplet.add_P1P2(
            *self.forward_chamfer(self.network, self.triplet.P2, self.triplet.P1, self.fix, True, 0, 0,
                                  self.distChamfer))
        self.triplet.add_P3P2(*self.forward_chamfer(self.network, self.triplet.P2, self.triplet.P3, self.fix, True,
                                                    self.triplet.latent_P2_1, 0, self.distChamfer))
        self.triplet.add_P2P3(*self.forward_chamfer(self.network, self.triplet.P3, self.triplet.P2, self.fix, True, 0,
                                                    self.triplet.latent_P2_2, self.distChamfer))
        self.triplet.add_P1P3(*self.forward_chamfer(self.network, self.triplet.P3, self.triplet.P1, self.fix, True,
                                                    self.triplet.latent_P3_1, self.triplet.latent_P1_2,
                                                    self.distChamfer))
        self.triplet.add_P3P1(*self.forward_chamfer(self.network, self.triplet.P1, self.triplet.P3, self.fix, True,
                                                    self.triplet.latent_P1_1, self.triplet.latent_P3_2,
                                                    self.distChamfer))
        self.triplet.compute_loss_train_Deformation_ChamferL2()

    def triplet_chamfer_loss_part_supervision(self):
        """
        Compute all reconstruction on a triplet P1,P2,P3 with part supervision
        """
        self.triplet.add_P2P1(
            *self.forward_chamfer_bypart(self.network, self.triplet.P1, self.triplet.label1, self.triplet.P2,
                                         self.triplet.label2, self.fix, True, 0, 0, self.distChamfer))
        self.triplet.add_P1P2(
            *self.forward_chamfer_bypart(self.network, self.triplet.P2, self.triplet.label2, self.triplet.P1,
                                         self.triplet.label1, self.fix, True, 0, 0, self.distChamfer))
        self.triplet.add_P3P2(
            *self.forward_chamfer_bypart(self.network, self.triplet.P2, self.triplet.label2, self.triplet.P3,
                                         self.triplet.label3, self.fix, True, self.triplet.latent_P2_1, 0,
                                         self.distChamfer))
        self.triplet.add_P2P3(
            *self.forward_chamfer_bypart(self.network, self.triplet.P3, self.triplet.label3, self.triplet.P2,
                                         self.triplet.label2, self.fix, True, 0, self.triplet.latent_P2_2,
                                         self.distChamfer))
        self.triplet.add_P1P3(
            *self.forward_chamfer_bypart(self.network, self.triplet.P3, self.triplet.label3, self.triplet.P1,
                                         self.triplet.label1, self.fix, True, self.triplet.latent_P3_1,
                                         self.triplet.latent_P1_2, self.distChamfer))
        self.triplet.add_P3P1(
            *self.forward_chamfer_bypart(self.network, self.triplet.P1, self.triplet.label1, self.triplet.P3,
                                         self.triplet.label3, self.fix, True, self.triplet.latent_P1_1,
                                         self.triplet.latent_P3_2, self.distChamfer))
        self.triplet.compute_loss_train_Deformation_ChamferL2_bypart()

    def train_iteration(self):
        self.optimizer.zero_grad()
        self.triplet.separate_points_normal_labels()

        # RECONSTRUCTION
        if self.opt.part_supervision == 0:
            self.triplet_chamfer_loss()
        else:
            self.triplet_chamfer_loss_part_supervision()

        loss_train_total = self.opt.lambda_chamfer * self.triplet.loss_train_Deformation_ChamferL2
        self.log.update("loss_train_Deformation_ChamferL2", self.triplet.loss_train_Deformation_ChamferL2)

        # CYCLE LENGHT 2
        if self.opt.lambda_cycle_2 != 0:
            self.triplet.compute_loss_cycle2()
            loss_train_total = loss_train_total + self.opt.lambda_cycle_2 * self.triplet.loss_train_cycleL2_2
            self.log.update("loss_train_cycleL2_2", self.triplet.loss_train_cycleL2_2)

        # CYCLE LENGHT 3
        if self.opt.lambda_cycle_3 != 0:
            self.triplet.compute_loss_cycle3()
            loss_train_total = loss_train_total + self.opt.lambda_cycle_3 * self.triplet.loss_train_cycleL2_3
            self.log.update("loss_train_cycleL2_3", self.triplet.loss_train_cycleL2_3)

        # SYNTHETIC DEFORMATION
        if self.epoch < self.opt.epoch_reconstruct:
            self.triplet.compute_loss_selfReconstruction(self.network)
            loss_train_total = loss_train_total + 1.0 * self.triplet.loss_train_selfReconstruction_L2
            self.log.update("loss_train_selfReconstruction_L2", self.triplet.loss_train_selfReconstruction_L2)

        elif self.opt.lambda_reconstruct != 0 and self.epoch > self.opt.epoch_reconstruct - 1:
            self.triplet.compute_loss_selfReconstruction(self.network)
            loss_train_total = loss_train_total + self.opt.lambda_reconstruct * self.triplet.loss_train_selfReconstruction_L2
            self.log.update("loss_train_selfReconstruction_L2", self.triplet.loss_train_selfReconstruction_L2)
        else:
            pass

        loss_train_total.backward()

        self.log.update("loss_train_total", loss_train_total)
        self.optimizer.step()  # gradient update

        # VIZUALIZE
        if self.iteration % 50 == 1 and self.opt.display:
            self.visualizer.show_pointclouds(points=self.triplet.P2[0], Y=self.triplet.label2[0], title="train_B")
            self.visualizer.show_pointclouds(points=self.triplet.P1[0], Y=self.triplet.label1[0], title="train_A")
            self.visualizer.show_pointclouds(points=self.triplet.P2_P1[0], Y=self.triplet.label1[0],
                                             title="train_B_reconstructed")

        self.print_iteration_stats(loss_train_total)

    def get_triplet(self, iterator):
        """
        Get a TripletPointCloud object to run a forward-backward iteration
        """
        P1, _, _, _, P2, _, _, _, P3, _, rot_matrix, _ = iterator.next()
        self.triplet = triplet_point_cloud.TripletPointCloud(P1=P1, P2=P2, P3=P3, rot_matrix=rot_matrix,
                                                             chamfer_loss_type=self.opt.chamfer_loss_type)

    def train_epoch(self):
        self.log.reset()
        self.network.train()
        self.learning_rate_scheduler()
        start = time.time()
        iterator = self.dataloader_train.__iter__()
        self.reset_iteration()
        while True:
            try:
                self.get_triplet(iterator)
                self.increment_iteration()
            except:
                print(colored("end of train dataset", 'red'))
                break
            self.train_iteration()
        print("Ellapsed time : ", time.time() - start)

    def test_iteration(self):
        self.couple.separate_points_normal_labels()
        batchs = self.couple.P1.size(0)
        self.couple.add_P2P1(*loss.forward_chamfer(self.network, self.couple.P1, self.couple.P2, local_fix=self.fix,
                                                   distChamfer=self.distChamfer))
        loss_val_Deformation_ChamferL2 = loss.chamferL2(self.couple.dist1_P2_P1, self.couple.dist2_P2_P1)
        loss_val_Reconstruction_L2 = loss.L2(self.couple.P2_P1, self.couple.P2)
        self.log.update("loss_val_Deformation_ChamferL2", loss_val_Deformation_ChamferL2)
        self.log.update("loss_val_Reconstruction_L2", loss_val_Reconstruction_L2)
        print(
            '\r' + colored('[%d: %d/%d]' % (self.epoch, self.iteration, self.len_dataset_test / (self.opt.batch_size)),
                           'red') +
            colored('loss_val_Deformation_ChamferL2:  %f' % loss_val_Deformation_ChamferL2.item(), 'yellow'),
            end='')

        if self.iteration % 60 == 1 and self.opt.display:
            self.visualizer.show_pointclouds(points=self.couple.P2[0], Y=self.couple.label2[0], title="val_B")
            self.visualizer.show_pointclouds(points=self.couple.P1[0], Y=self.couple.label1[0], title="val_A")
            self.visualizer.show_pointclouds(points=self.couple.P2_P1[0], Y=self.couple.label1[0],
                                             title="val_B_reconstructed")

        # Compute Miou when labels are tranfered from P1 to P2.
        predicted_target = self.couple.label1.view(-1)[self.couple.idx2_P2_P1].view(batchs, -1)
        for shape in range(batchs):
            if self.couple.cat_1 == self.couple.cat_2:
                target = self.couple.label2[shape].squeeze().data.cpu().numpy()
                iou_val = miou_shape.miou_shape(predicted_target[shape].squeeze().cpu().numpy(), target,
                                                self.dataset_train.part_category[self.couple.cat_1[shape]])
                self.log.update("iou_val", iou_val)

    def get_couple(self, iterator):
        """
        Get a TripletPointCloud object to run a forward-backward iteration
        """
        P1, cat_1, _, _, P2, cat_2, _, _, P3, _, rot_matrix, _ = iterator.next()
        self.couple = couple_point_cloud.CouplePointCloud(P1=P1, P2=P2, cat_1=cat_1, cat_2=cat_2)

    def test_epoch(self):
        self.network.eval()
        iterator = self.dataloader_test.__iter__()
        self.reset_iteration()
        while True:
            self.increment_iteration()
            try:
                self.get_couple(iterator)
            except:
                print(colored("end of val dataset", 'red'))
                break
            self.test_iteration()

        self.log.end_epoch()
        if self.opt.display:
            self.log.update_curves(self.visualizer.vis, self.opt.save_path)

