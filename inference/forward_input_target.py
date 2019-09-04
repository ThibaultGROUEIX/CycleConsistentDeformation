import sys

"""
USAGE
python ./inference/get_criterion_shape.py \
     --eval_target ./data/dataset_shapenet/02958343/731efc7a52841a5a59139efcde1fedcb.txt \
     --eval_source ./data/dataset_shapenet/02958343/c3858a8b73dcb137e3bdba9430565083.txt \
     --logdir Car_unsup \
     --shapenetv1_path /trainman-mount/trainman-storage-e7719e4d-b36c-4bc0-a3b3-e13a2d53f66d/ShapeNetCore.v1 \
"""
sys.path.append("./auxiliary/")
sys.path.append("./extension/")
sys.path.append("./training/")
sys.path.append("./inference/")
sys.path.append("./scripts/")
sys.path.append("./")
import argument_parser
import trainer as t
import useful_losses as loss
import my_utils
from save_mesh_from_points_and_labels import *
import pymesh
import get_shapenet_model
import high_frequencies
import torch
import figure_2_3
import os


def forward(opt):
    """
    Takes an input and a target mesh. Deform input in output and propagate a
    manually defined high frequency from the oinput to the output
    :return:
    """
    my_utils.plant_seeds(randomized_seed=opt.randomize)

    trainer = t.Trainer(opt)
    trainer.build_dataset_train_for_matching()
    trainer.build_dataset_test_for_matching()
    trainer.build_network()
    trainer.build_losses()
    trainer.network.eval()

    if opt.eval_source[-4:] == ".txt":
        opt.eval_source = figure_2_3.convert_path(opt.shapenetv1_path, opt.eval_source)
    if opt.eval_target[-4:] == ".txt":
        opt.eval_target = figure_2_3.convert_path(opt.shapenetv1_path, opt.eval_target)

    path_deformed = os.path.join("./figures/forward_input_target/", opt.eval_source[-42:-10] + "deformed.ply")
    path_source = os.path.join("./figures/forward_input_target/", opt.eval_source[-42:-10] + ".ply")
    path_target = os.path.join("./figures/forward_input_target/",
                               opt.eval_source[-42:-10] + "_" + opt.eval_target[-42:-10] + ".ply")

    mesh_path = opt.eval_source
    print(mesh_path)
    source_mesh_edge = get_shapenet_model.link(mesh_path)

    mesh_path = opt.eval_target
    target_mesh_edge = get_shapenet_model.link(mesh_path)

    pymesh.save_mesh(path_source, source_mesh_edge, ascii=True)
    pymesh.save_mesh(path_target, target_mesh_edge, ascii=True)
    print("Deforming source in target")

    source = torch.from_numpy(source_mesh_edge.vertices).cuda().float().unsqueeze(0)
    target = torch.from_numpy(target_mesh_edge.vertices).cuda().float().unsqueeze(0)

    with torch.no_grad():
        source, _, _, _, _ = loss.forward_chamfer(trainer.network, source, target, local_fix=None,
                                                  distChamfer=trainer.distChamfer)

    P2_P1_mesh = pymesh.form_mesh(vertices=source.squeeze().cpu().detach().numpy(), faces=source_mesh_edge.faces)
    pymesh.save_mesh(path_deformed, P2_P1_mesh, ascii=True)

    print("computing signal tranfer form source to target")
    high_frequencies.high_frequency_propagation(path_source, path_deformed, path_target)


if __name__ == '__main__':
    opt = argument_parser.parser()
    forward(opt)
