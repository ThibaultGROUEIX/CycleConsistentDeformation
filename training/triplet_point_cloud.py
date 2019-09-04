import torch
import useful_losses as loss

class TripletPointCloud:
    """
    This class stores a triplet of point P1, P2, P3 and the various attributes useful to compute the associated losses.
    """
    def __init__(self, P1, P2, P3, rot_matrix=None, cat_1=None, cat_2=None, cat_3=None, chamfer_loss_type="SYM"):
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3
        self.cat_1 = cat_1
        self.cat_2 = cat_2
        self.cat_3 = cat_3
        self.rot_matrix = rot_matrix
        self.loss_train_Deformation_ChamferL2 = 0
        self.chamfer_loss = loss.ChamferLoss(loss_type = chamfer_loss_type, verbose=False)

    def separate_points_normal_labels(self):
        self.label1 = self.P1[:, :, 6].contiguous()
        self.label2 = self.P2[:, :, 6].contiguous()
        self.label3 = self.P3[:, :, 6].contiguous()
        self.batchs = self.P1.size(0)

        self.P1 = self.P1[:, :, :3].contiguous().cuda().float()
        self.P2 = self.P2[:, :, :3].contiguous().cuda().float()
        self.P3 = self.P3[:, :, :3].contiguous().cuda().float()


    def add_P2P1(self, P2_P1, dist1_P2_P1, dist2_P2_P1, idx1_P2_P1, idx2_P2_P1, latent_P1_1, latent_P2_2, loss_P2_P1=0):
        self.loss_P2_P1 = loss_P2_P1
        self.P2_P1 = P2_P1
        self.dist1_P2_P1 = dist1_P2_P1
        self.dist2_P2_P1 = dist2_P2_P1
        self.idx1_P2_P1 = idx1_P2_P1
        self.idx2_P2_P1 = idx2_P2_P1
        self.latent_P1_1 = latent_P1_1
        self.latent_P2_2 = latent_P2_2

    def add_P1P2(self, P1_P2, dist1_P1_P2, dist2_P1_P2, idx1_P1_P2, idx2_P1_P2, latent_P2_1, latent_P1_2, loss_P1_P2=0):
        self.loss_P1_P2 = loss_P1_P2
        self.P1_P2 = P1_P2
        self.dist1_P1_P2 = dist1_P1_P2
        self.dist2_P1_P2 = dist2_P1_P2
        self.idx1_P1_P2 = idx1_P1_P2
        self.idx2_P1_P2 = idx2_P1_P2
        self.latent_P2_1 = latent_P2_1
        self.latent_P1_2 = latent_P1_2

    def add_P3P2(self, P3_P2, dist1_P3_P2, dist2_P3_P2, idx1_P3_P2, idx2_P3_P2, latent_P2_1, latent_P3_2, loss_P3_P2=0):
        self.loss_P3_P2 = loss_P3_P2
        self.P3_P2 = P3_P2
        self.dist1_P3_P2 = dist1_P3_P2
        self.dist2_P3_P2 = dist2_P3_P2
        self.idx1_P3_P2 = idx1_P3_P2
        self.idx2_P3_P2 = idx2_P3_P2
        self.latent_P2_1 = latent_P2_1
        self.latent_P3_2 = latent_P3_2

    def add_P2P3(self, P2_P3, dist1_P2_P3, dist2_P2_P3, idx1_P2_P3, idx2_P2_P3, latent_P3_1, latent_P2_2, loss_P2_P3=0):
        self.loss_P2_P3 = loss_P2_P3
        self.P2_P3 = P2_P3
        self.dist1_P2_P3 = dist1_P2_P3
        self.dist2_P2_P3 = dist2_P2_P3
        self.idx1_P2_P3 = idx1_P2_P3
        self.idx2_P2_P3 = idx2_P2_P3
        self.latent_P3_1 = latent_P3_1
        self.latent_P2_2 = latent_P2_2

    def add_P1P3(self, P1_P3, dist1_P1_P3, dist2_P1_P3, idx1_P1_P3, idx2_P1_P3, latent_P3_1, latent_P1_2, loss_P1_P3=0):
        self.loss_P1_P3 = loss_P1_P3
        self.P1_P3 = P1_P3
        self.dist1_P1_P3 = dist1_P1_P3
        self.dist2_P1_P3 = dist2_P1_P3
        self.idx1_P1_P3 = idx1_P1_P3
        self.idx2_P1_P3 = idx2_P1_P3
        self.latent_P3_1 = latent_P3_1
        self.latent_P1_2 = latent_P1_2

    def add_P3P1(self, P3_P1, dist1_P3_P1, dist2_P3_P1, idx1_P3_P1, idx2_P3_P1, latent_P1_1, latent_P3_2, loss_P3_P1=0):
        self.loss_P3_P1 = loss_P3_P1
        self.P3_P1 = P3_P1
        self.dist1_P3_P1 = dist1_P3_P1
        self.dist2_P3_P1 = dist2_P3_P1
        self.idx1_P3_P1 = idx1_P3_P1
        self.idx2_P3_P1 = idx2_P3_P1
        self.latent_P1_1 = latent_P1_1
        self.latent_P3_2 = latent_P3_2

    def compute_loss_train_Deformation_ChamferL2(self):
        self.loss_train_Deformation_ChamferL2 = (1 / 6.0) * (
                self.chamfer_loss.forward(self.dist1_P2_P1, self.dist2_P2_P1) +
                self.chamfer_loss.forward(self.dist1_P1_P2, self.dist2_P1_P2) +
                self.chamfer_loss.forward(self.dist1_P3_P2, self.dist2_P3_P2) +
                self.chamfer_loss.forward(self.dist1_P2_P3, self.dist2_P2_P3) +
                self.chamfer_loss.forward(self.dist1_P1_P3, self.dist2_P1_P3) +
                self.chamfer_loss.forward(self.dist1_P3_P1, self.dist2_P3_P1)
        )

    def compute_loss_train_Deformation_ChamferL2_bypart(self):
        self.loss_train_Deformation_ChamferL2  = (1 / 6.0) * (
                        self.loss_P2_P1 + self.loss_P1_P2 + self.loss_P3_P2 + self.loss_P2_P3 + self.loss_P1_P3 + self.loss_P3_P1)


    def compute_loss_cycle2(self):
        P1_P2_cycle = loss.batch_cycle_2(self.P1_P2, self.idx1_P2_P1, self.batchs)
        P2_P1_cycle = loss.batch_cycle_2(self.P2_P1, self.idx1_P1_P2, self.batchs)
        P3_P2_cycle = loss.batch_cycle_2(self.P3_P2, self.idx1_P2_P3, self.batchs)
        P2_P3_cycle = loss.batch_cycle_2(self.P2_P3, self.idx1_P3_P2, self.batchs)
        P1_P3_cycle = loss.batch_cycle_2(self.P1_P3, self.idx1_P3_P1, self.batchs)
        P3_P1_cycle = loss.batch_cycle_2(self.P3_P1, self.idx1_P1_P3, self.batchs)
        self.loss_train_cycleL2_2 = (1 / 6.0) * (
                    loss.L2(P1_P2_cycle, self.P1) + loss.L2(P2_P1_cycle, self.P2) + loss.L2(P3_P2_cycle, self.P3) + loss.L2(P2_P3_cycle, self.P2) + loss.L2(
                P1_P3_cycle, self.P1) + loss.L2(P3_P1_cycle, self.P3))

    def compute_loss_cycle3(self):
        P1_P3_P2_cycle = loss.batch_cycle_3(self.P1_P3, self.idx1_P2_P1, self.idx1_P3_P2, self.batchs)
        P2_P1_P3_cycle = loss.batch_cycle_3(self.P2_P1, self.idx1_P3_P2, self.idx1_P1_P3, self.batchs)
        P3_P2_P1_cycle = loss.batch_cycle_3(self.P3_P2, self.idx1_P1_P3, self.idx1_P2_P1, self.batchs)
        P1_P2_P3_cycle = loss.batch_cycle_3(self.P1_P2, self.idx1_P3_P1, self.idx1_P2_P3, self.batchs)
        P2_P3_P1_cycle = loss.batch_cycle_3(self.P2_P3, self.idx1_P1_P2, self.idx1_P3_P1, self.batchs)
        P3_P1_P2_cycle = loss.batch_cycle_3(self.P3_P1, self.idx1_P2_P3, self.idx1_P1_P2, self.batchs)
        self.loss_train_cycleL2_3 = (1 / 6.0) * (
                    loss.L2(P1_P3_P2_cycle, self.P1) + loss.L2(P2_P1_P3_cycle, self.P2) + loss.L2(P3_P2_P1_cycle, self.P3) + loss.L2(P1_P2_P3_cycle,
                                                                                                  self.P1) + loss.L2(
                P2_P3_P1_cycle, self.P2) + loss.L2(P3_P1_P2_cycle, self.P3))

    def compute_loss_selfReconstruction(self, network):
        rot_matrix = self.rot_matrix.cuda().float().transpose(2,1).contiguous()

        P4 = self.P3.bmm(rot_matrix).detach()
        scale = torch.rand(P4.size(0), 1, 3).cuda() - 0.5  # uniform sampling -0.5, 0.5
        P4 = P4 + scale*P4 # Anisotropic scaling

        P3 = self.P3.transpose(2, 1).contiguous()
        P4 = P4.transpose(2, 1).contiguous()
        P4_P3 = network(P3, P4)  # forward pass
        P3_P4 = network(P4, P3)  # forward pass
        self.loss_train_selfReconstruction_L2 = (1 / 2.0) * (loss.L2(P4_P3, P4) + loss.L2(P3_P4, P3))

