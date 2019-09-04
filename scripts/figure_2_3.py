import get_shapenet_model
import pymesh
import high_frequencies
import save_mesh_from_points_and_labels as smfpl
import torch
import useful_losses as loss
import os


def convert_path(path_shapenet, name ):
    return os.path.join(path_shapenet, name[24:-4] + '/model.obj')

class MakeFigure(object):
    def __init__(self, opt, top_cycles, max_k_fig, min_k_fig, category, path_train_list, points_train_list,
                 labels_train_list, trainer, parts, distChamfer):
        self.opt = opt
        self.top_cycles = top_cycles
        self.max_k_fig = max_k_fig
        self.min_k_fig = min_k_fig
        self.category = category
        self.path_train_list = path_train_list
        self.points_train_list = points_train_list
        self.labels_train_list = labels_train_list
        self.trainer = trainer
        self.parts = parts
        self.distChamfer = distChamfer

    def deform_source_in_target(self, i, top, name):
        """

        :param i: ranking of  source,target pair in top_cycle
        :param top: index of  source,target pair in top_cycle
        :return: save meshes of deformation and label transfer
        top_cycles: {
            idx : idx of best train sammple for target
            values : value of cycle criterion for source/target pair
            iou_ours : iou obtained with our method for source/target pairs
            iou_NN : iou of the nearest neighbors for source/target pairs
            predicted_NN_P2_P0 : label indexes of target with the nearest neighbors for source/target pairs
            predicted_ours_P2_P0 : label indexes of target with our methodfor source/target pairs
            P2_P0 : deformed source in target
            P2 : target
            P2_label : target original labels
            P2_path : Path to target
        """
        idx, values, iou_ours, iou_NN, predicted_NN_P2_P0, predicted_ours_P2_P0, P2_P0, P2, P2_label, P2_path = \
        self.top_cycles[top]
        print("computing results for ...", P2_path)
        train_path = self.path_train_list[idx]
        mesh_path = convert_path(self.opt.shapenetv1_path, train_path)
        source_mesh_edge = get_shapenet_model.link(mesh_path)
        pymesh.save_mesh(
            os.path.join(self.figure_folder, name + "_" + str(i) + "_SourceMesh_" + train_path[33:-4] + ".ply"),
            source_mesh_edge, ascii=True)

        mesh_path = convert_path(self.opt.shapenetv1_path, P2_path)
        target_mesh_edge = get_shapenet_model.link(mesh_path)
        pymesh.save_mesh(
            os.path.join(self.figure_folder, name + "_" + str(i) + "_TargetMesh_" + P2_path[33:-4] + ".ply"),
            target_mesh_edge, ascii=True)

        with torch.no_grad():
            P2_P1_mesh, _, _, _, _ = loss.forward_chamfer(self.trainer.network, torch.from_numpy(
                source_mesh_edge.vertices).cuda().float().unsqueeze(0), torch.from_numpy(
                target_mesh_edge.vertices).cuda().float().unsqueeze(0), local_fix=None, distChamfer=self.distChamfer)
        P2_P1_mesh = pymesh.form_mesh(vertices=P2_P1_mesh.squeeze().cpu().numpy(), faces=source_mesh_edge.faces)
        pymesh.save_mesh(
            os.path.join(self.figure_folder, name + "_" + str(i) + "_SourceMeshDeformed_" + train_path[33:-4] + ".ply"),
            P2_P1_mesh, ascii=True)

        high_frequencies.high_frequency_propagation(
            os.path.join(self.figure_folder, name + "_" + str(i) + "_SourceMesh_" + train_path[33:-4] + ".ply"),
            os.path.join(self.figure_folder, name + "_" + str(i) + "_SourceMeshDeformed_" + train_path[33:-4] + ".ply"),
            os.path.join(self.figure_folder, name + "_" + str(i) + "_TargetMesh_" + P2_path[33:-4] + ".ply"))

        smfpl.save_mesh_from_pointsandlabels(self.points_train_list[idx], self.labels_train_list[idx],
                                             os.path.join(self.figure_folder,
                                                          name + "_" + str(i) + "_SourcePoints_" + train_path[
                                                                                                   33:-4] + ".ply"),
                                             parts=self.parts)
        smfpl.save_mesh_from_pointsandlabels(P2, P2_label,
                                             os.path.join(self.figure_folder,
                                                          name + "_" + str(i) + "_TargetPoints_" + P2_path[
                                                                                                   33:-4] + ".ply"),
                                             parts=self.parts)
        smfpl.save_mesh_from_pointsandlabels(P2_P0[0], self.labels_train_list[idx],
                                             os.path.join(self.figure_folder,
                                                          name + "_" + str(i) + "_SourcePointsDeformed_" + P2_path[
                                                                                                           33:-4] + ".ply"),
                                             parts=self.parts)
        smfpl.save_mesh_from_pointsandlabels(P2, predicted_ours_P2_P0,
                                             os.path.join(self.figure_folder, name + "_" + str(
                                                 i) + "_TargetPointsPredictedLabelsOURS_" + P2_path[
                                                                                            33:-4] + ".ply"),
                                             parts=self.parts)
        smfpl.save_mesh_from_pointsandlabels(P2, predicted_NN_P2_P0,
                                             os.path.join(self.figure_folder, name + "_" + str(
                                                 i) + "_TargetPointsPredictedLabelsNN_" + P2_path[
                                                                                          33:-4] + ".ply"),
                                             parts=self.parts)
        with open(os.path.join(self.figure_folder, name + "_" + str(i) + "_stats.txt"), 'w') as fp:
            fp.write(f"index {name} train sample : " + str(idx) + '\n')
            fp.write(f"value {name} train sample : " + '%.2f' % (values * 100) + '\n')
            fp.write(f"iou_ours {name} train sample : " + '%.2f' % (iou_ours * 100) + '\n')
            fp.write(f"iou_NN {name} train sample : " + '%.2f' % (iou_NN * 100) + '\n')

    def make_figure_2_3(self):
        # Best Cycle criterion
        top_cycles_val = list(map(lambda x: x[1], self.top_cycles))
        top_k_idx, top_k_values = self.max_k_fig(top_cycles_val)
        min_k_idx, min_k_values = self.min_k_fig(top_cycles_val)
        self.figure_folder = os.path.join("figures", self.category + "_figure_high_freq")

        if not os.path.exists(self.figure_folder):
            print("creating folder  ", self.figure_folder)
            os.mkdir(self.figure_folder)

        for i, top in enumerate(top_k_idx):
            self.deform_source_in_target(i, top, "worseCycle")

        for i, top in enumerate(min_k_idx):
            self.deform_source_in_target(i, top, "bestCycle")

        # Biggest difference to Nearest Neighbors to get the interesting cases
        top_cycles_val = list(map(lambda x: x[3] - x[2], self.top_cycles))  # nous - eux
        top_k_idx, top_k_values = self.max_k_fig(top_cycles_val)
        min_k_idx, min_k_values = self.min_k_fig(top_cycles_val)

        for i, top in enumerate(min_k_idx):
            self.deform_source_in_target(i, top, "worseComparison")

        for i, top in enumerate(top_k_idx):
            self.deform_source_in_target(i, top, "bestComparison")

        # IoU propagation to see working and failing propagations
        top_cycles_val = list(map(lambda x: x[2], self.top_cycles))
        top_k_idx, top_k_values = self.max_k_fig(top_cycles_val)
        min_k_idx, min_k_values = self.min_k_fig(top_cycles_val)

        for i, top in enumerate(top_k_idx):
            self.deform_source_in_target(i, top, "bestIOU")

        for i, top in enumerate(min_k_idx):
            self.deform_source_in_target(i, top, "worseIOU")
