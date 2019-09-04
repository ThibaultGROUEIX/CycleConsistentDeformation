from __future__ import print_function
"""
USAGE
python ./inference/get_criterion_shape.py \
     --eval_get_criterions_for_shape ./data/dataset_shapenet/02958343/731efc7a52841a5a59139efcde1fedcb.txt \
     --logdir Car_unsup \
     --shapenetv1_path /trainman-mount/trainman-storage-e7719e4d-b36c-4bc0-a3b3-e13a2d53f66d/ShapeNetCore.v1 \
     --cat Car \
"""
import sys
sys.path.append("./auxiliary/")
sys.path.append("./extension/")
sys.path.append("./training/")
sys.path.append("./inference/")
sys.path.append("./")
import argument_parser
from auxiliary.utils import *
from auxiliary.model_atlasnet import *
import miou_shape
import useful_losses as loss
import my_utils
from my_utils import Max_k, Min_k
import get_shapenet_model
import normalize_points
import trainer as t
import pprint
import forward_source_target

def get_criterion_shape(opt):
    return_dict = {}
    my_utils.plant_seeds(randomized_seed=opt.randomize)

    trainer = t.Trainer(opt)
    trainer.build_dataset_train_for_matching()
    trainer.build_dataset_test_for_matching()
    trainer.build_network()
    trainer.build_losses()
    trainer.network.eval()

    # Load input mesh
    exist_P2_label = True

    try:
        mesh_path = opt.eval_get_criterions_for_shape  # Ends in .txt
        points = np.loadtxt(mesh_path)
        points = torch.from_numpy(points).float()
        # Normalization is done before resampling !
        P2 = normalize_points.BoundingBox(points[:, :3])
        P2_label = points[:, 6].data.cpu().numpy()
    except:
        mesh_path = opt.eval_get_criterions_for_shape  # Ends in .obj
        source_mesh_edge = get_shapenet_model.link(mesh_path)
        P2 = torch.from_numpy(source_mesh_edge.vertices)
        exist_P2_label = False

    min_k = Min_k(opt.k_max_eval)
    max_k = Max_k(opt.k_max_eval)

    points_train_list = []
    point_train_paths = []
    labels_train_list = []
    iterator_train = trainer.dataloader_train.__iter__()

    for find_best in range(opt.num_shots_eval):
        try:
            points_train, _, _, file_path = iterator_train.next()
            points_train_list.append(points_train[:, :, :3].contiguous().cuda().float())
            point_train_paths.append(file_path)
            labels_train_list.append(points_train[:, :, 6].contiguous().cuda().float())
        except:
            break

    # ========Loop on test examples======================== #
    with torch.no_grad():
        P2 = P2[:, :3].unsqueeze(0).contiguous().cuda().float()
        P2_latent = trainer.network.encode(P2.transpose(1, 2).contiguous(), P2.transpose(1, 2).contiguous())

        # Chamfer (P0_P2)
        P0_P2_list = list(
            map(lambda x: loss.forward_chamfer(trainer.network, P2, x, local_fix=None, distChamfer=trainer.distChamfer), points_train_list))

        # Compute Chamfer (P2_P0)
        P2_P0_list = list(
            map(lambda x: loss.forward_chamfer(trainer.network, x, P2, local_fix=None, distChamfer=trainer.distChamfer), points_train_list))

        predicted_ours_P2_P0_list = list(
            map(lambda x, y: x.view(-1)[y[4].view(-1).data.long()].view(1, -1), labels_train_list, P2_P0_list))

        if exist_P2_label:
            iou_ours_list = list(map(lambda x: miou_shape.miou_shape(x.squeeze().cpu().numpy(), P2_label, trainer.parts),
                                     predicted_ours_P2_P0_list))
            top_k_idx, top_k_values = max_k(iou_ours_list)
            return_dict["oracle"] = point_train_paths[top_k_idx[0]][0]

        predicted_ours_P2_P0_list = list(
            map(lambda x, y: x.view(-1)[y[4].view(-1).data.long()].view(1, -1), labels_train_list, P2_P0_list))
        predicted_ours_P2_P0_list = torch.cat(predicted_ours_P2_P0_list)

        # Compute NN
        P2_P0_NN_list = list(map(lambda x: loss.distChamfer(x, P2), points_train_list))
        predicted_NN_P2_P0_list = list(
            map(lambda x, y: x.view(-1)[y[3].view(-1).data.long()].view(1, -1), labels_train_list, P2_P0_NN_list))
        predicted_NN_P2_P0_list = torch.cat(predicted_NN_P2_P0_list)

        # NN
        NN_chamferL2_list = list(map(lambda x: loss.chamferL2(x[0], x[1]), P2_P0_NN_list))
        top_k_idx, top_k_values = min_k(NN_chamferL2_list)
        return_dict["NN_criterion"] = point_train_paths[top_k_idx[0]][0]

        # Chamfer ours
        chamfer_list = list(map(lambda x: loss.chamferL2(x[1], x[2]), P2_P0_list))
        top_k_idx, top_k_values = min_k(chamfer_list)
        return_dict["chamfer_criterion"] = point_train_paths[top_k_idx[0]][0]

        # NN in latent space
        P0_latent_list = list(
            map(lambda x: trainer.network.encode(x.transpose(1, 2).contiguous(), x.transpose(1, 2).contiguous()),
                points_train_list))
        cosine_list = list(map(lambda x: loss.cosine(x, P2_latent), P0_latent_list))

        top_k_idx, top_k_values = min_k(cosine_list)
        return_dict["cosine_criterion"] = point_train_paths[top_k_idx[0]][0]

        # Cycle 2
        P0_P2_cycle_list = list(map(lambda x, y: loss.batch_cycle_2(x[0], y[3], 1), P0_P2_list, P2_P0_list))
        P0_P2_cycle_list = list(map(lambda x, y: loss.L2(x, y), P0_P2_cycle_list, points_train_list))

        P2_P0_cycle_list = list(map(lambda x, y: loss.batch_cycle_2(x[0], y[3], 1), P2_P0_list, P0_P2_list))
        P2_P0_cycle_list = list(map(lambda x: loss.L2(x, P2), P2_P0_cycle_list))

        # Cycle 2 both sides
        both_cycle_list = list(map(lambda x, y: x * y, P0_P2_cycle_list, P2_P0_cycle_list))
        both_cycle_list = np.power(both_cycle_list, 1.0 / 2.0).tolist()
        top_k_cycle2_idx, top_k_values = min_k(both_cycle_list)
        return_dict["cycle_criterion"] = point_train_paths[top_k_cycle2_idx[0]][0]
        pprint.pprint(return_dict)
        return return_dict


if __name__ == '__main__':
    opt = argument_parser.parser()
    return_dict = get_criterion_shape(opt)
    for key in return_dict:
        opt.eval_source = return_dict[key]
        opt.eval_target = opt.eval_get_criterions_for_shape
        forward_source_target.forward(opt)
    pprint.pprint(return_dict)
