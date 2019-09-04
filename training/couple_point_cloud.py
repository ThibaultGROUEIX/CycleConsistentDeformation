import useful_losses as loss

class CouplePointCloud:
    """
    This class stores a couple of point P1, P2 and the various attributes useful to compute the associated losses.
    """
    def __init__(self, P1, P2, rot_matrix=None, cat_1=None, cat_2=None):
        self.P1 = P1
        self.P2 = P2
        self.cat_1 = cat_1
        self.cat_2 = cat_2
        self.rot_matrix = rot_matrix
        self.loss_train_Deformation_ChamferL2 = 0


    def separate_points_normal_labels(self):
        self.label1 = self.P1[:, :, 6].contiguous()
        self.label2 = self.P2[:, :, 6].contiguous()
        self.batchs = self.P1.size(0)

        self.P1 = self.P1[:, :, :3].contiguous().cuda().float()
        self.P2 = self.P2[:, :, :3].contiguous().cuda().float()


    def add_P2P1(self, P2_P1, dist1_P2_P1, dist2_P2_P1, idx1_P2_P1, idx2_P2_P1, latent_P1_1 = None, latent_P2_2 = None, loss_P2_P1=0):
        self.loss_P2_P1 = loss_P2_P1
        self.P2_P1 = P2_P1
        self.dist1_P2_P1 = dist1_P2_P1
        self.dist2_P2_P1 = dist2_P2_P1
        self.idx1_P2_P1 = idx1_P2_P1
        self.idx2_P2_P1 = idx2_P2_P1
        self.latent_P1_1 = latent_P1_1
        self.latent_P2_2 = latent_P2_2

    def add_P1P2(self, P1_P2, dist1_P1_P2, dist2_P1_P2, idx1_P1_P2, idx2_P1_P2, latent_P2_1 = None, latent_P1_2 = None, loss_P1_P2=0):
        self.loss_P1_P2 = loss_P1_P2
        self.P1_P2 = P1_P2
        self.dist1_P1_P2 = dist1_P1_P2
        self.dist2_P1_P2 = dist2_P1_P2
        self.idx1_P1_P2 = idx1_P1_P2
        self.idx2_P1_P2 = idx2_P1_P2
        self.latent_P2_1 = latent_P2_1
        self.latent_P1_2 = latent_P1_2


    def compute_loss_train_Deformation_ChamferL2(self):
        self.loss_train_Deformation_ChamferL2 = (1 / 2.0) * (
                loss.chamferL2(self.dist1_P2_P1, self.dist2_P2_P1) + loss.chamferL2(self.dist1_P1_P2, self.dist2_P1_P2))

    def compute_loss_train_Deformation_ChamferL2_bypart(self):
        self.loss_train_Deformation_ChamferL2  = (1 / 2.0) * (self.loss_P2_P1 + self.loss_P1_P2 )


    def compute_loss_cycle2(self):
        P1_P2_cycle = loss.batch_cycle_2(self.P1_P2, self.idx1_P2_P1, self.batchs)
        P2_P1_cycle = loss.batch_cycle_2(self.P2_P1, self.idx1_P1_P2, self.batchs)

        self.loss_train_cycleL2_2 = (1 / 6.0) * (
                    loss.L2(P1_P2_cycle, self.P1) + loss.L2(P2_P1_cycle, self.P2) )

    def compute_loss_cycle3(self):
        P1_P3_P2_cycle = loss.batch_cycle_3(self.P1_P3, self.idx1_P2_P1, self.idx1_P3_P2, self.batchs)
        P2_P1_P3_cycle = loss.batch_cycle_3(self.P2_P1, self.idx1_P3_P2, self.idx1_P1_P3, self.batchs)
        self.loss_train_cycleL2_3 = (1 / 6.0) * (loss.L2(P1_P3_P2_cycle, self.P1) + loss.L2(P2_P1_P3_cycle, self.P2))
