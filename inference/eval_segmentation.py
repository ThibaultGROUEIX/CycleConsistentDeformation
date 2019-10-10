from __future__ import print_function
import sys

sys.path.append("./auxiliary/")
sys.path.append("./extension/")
sys.path.append("./trainer/")
sys.path.append("./scripts/")
sys.path.append("./inference/")
sys.path.append("./training/")
sys.path.append("./")
import argument_parser
import trainer
from auxiliary.my_utils import *
from functools import reduce
from auxiliary.model_atlasnet import *
import miou_shape
import useful_losses as loss
import my_utils
from my_utils import Max_k, Min_k
import ICP
import tqdm
from save_mesh_from_points_and_labels import *
import figure_2_3

opt = argument_parser.parser()

my_utils.plant_seeds(randomized_seed=opt.randomize)
min_k = Min_k(opt.k_max_eval)
max_k = Max_k(opt.k_max_eval)
if opt.num_figure_3_4 > 0:
    min_k_fig = Min_k(opt.num_figure_3_4)
    max_k_fig = Max_k(opt.num_figure_3_4)

trainer = trainer.Trainer(opt)
trainer.build_dataset_train_for_matching()
trainer.build_dataset_test_for_matching()
trainer.build_network()
trainer.network.eval()

# =============DEFINE Criterions to evaluate======================================== #
NN_latent_space = True
crit = [
    "iou_NN",
    "iou_NN_ICP",
    "iou_chamfer",
    "iou_cosine",
    "iou_affine",
    "both_cycle",
    "oracle_NN",
    "oracle_ours",
    "oracle_atlasPatch",
    "oracle_atlasSphere",
]
criterions = []
for criterion in crit:
    criterions = criterions + [criterion + "_NN"]
    criterions = criterions + [criterion + "_ours"]
    criterions = criterions + [criterion + "_atlasSphere"]
    criterions = criterions + [criterion + "_atlasPatch"]
ens = []
for criterion in criterions:
    ens = ens + [criterion + "_emsemble"]
criterions = criterions + ens

global_meter = [AverageValueMeter() for key in criterions]
global_str = ["" for key in criterions]
top_cycles = []


# ========================================================== #

def update_meter(meter, name):
    try:
        meter.update(iou_dict[name])
    except:
        pass
        # print("No value for :"  + name)


def get_update_global_meter(num):
    def update_global_meter(global_meter, local_meter):
        try:
            global_meter.update(local_meter.avg, num)
        except:
            pass
            # print("No value for :"  + name)

    return update_global_meter


def print_avg(x, meter, name):
    try:
        # if meter.avg >0.01: #print only for computed meter
        #     print(name + " AVG: "  + + '%.2f' % (meter.avg*100))
        return x + " & " + '%.2f' % (meter.avg * 100)
    except:
        print("No avg for :" + name)


def print_global_avg(x, meter, name):
    try:
        if meter.avg > 0.01:  # print only for computed meter
            print(name + " Global AVG: " + '%.2f' % (meter.avg * 100))
        return name + " & " + '%.2f' % (meter.avg * 100) + " & " + x
    except:
        print("No avg for :" + name)


# =============DEFINE CHAMFER LOSS======================================== #
import extension.dist_chamfer_idx as ext

distChamfer = ext.chamferDist()
# ========================================================== #

# =============DEFINE Atlasnet Networks======================================== #
use_atlas_baseline = False
network_atlasSphere = AE_AtlasNet_SPHERE(num_points=5000, fixed_points=True)
network_atlasSphere.cuda()
if opt.atlasSphere_path != '':
    network_atlasSphere.load_state_dict(torch.load(opt.atlasSphere_path))
    print("reloaded atlasnet weights")
    print("Enable Atlasnet Baseline")
    use_atlas_baseline = True
network_atlasSphere.eval()

network_atlasPatch = AE_AtlasNet(num_points=5000, fixed_points=True)
network_atlasPatch.cuda()

if opt.atlasPatch_path != '':
    network_atlasPatch.load_state_dict(torch.load(opt.atlasPatch_path))
    print("reloaded atlasnet weights")
    use_atlas_baseline = True
    print("Enable Atlasnet Baseline")
network_atlasPatch.eval()
# ========================================================== #

for category in opt.categories:
    # meters to record stats on learning
    local_meter = [AverageValueMeter() for key in criterions]

    points_train_list = []
    labels_train_list = []
    path_train_list = []

    iterator_train = trainer.dataloader_train.__iter__()

    # Create source dataset os size num_shots_eval to select a labeled example from for each target
    for find_best in range(opt.num_shots_eval):
        try:
            points_train, _, _, file_path = iterator_train.next()
            points_train_list.append(points_train[:, :, :3].contiguous().cuda().float())
            labels_train_list.append(points_train[:, :, 6].contiguous().cuda().float())
            path_train_list.append(file_path[0])

        except:
            break

    # Reconstruct each source dataset samples with atlasnet
    if use_atlas_baseline:
        with torch.no_grad():
            atlasSphere_recons = list(
                map(lambda x: loss.forward_chamfer_atlasnet(network_atlasSphere, x, distChamfer=distChamfer),
                    points_train_list))
            atlasSphere_labels = list(
                map(lambda x, y: x.view(-1)[y[3].view(-1).data.long()].view(1, -1), labels_train_list,
                    atlasSphere_recons))
            atlasPatch_recons = list(
                map(lambda x: loss.forward_chamfer_atlasnet(network_atlasPatch, x, distChamfer=distChamfer),
                    points_train_list))
            atlasPatch_labels = list(
                map(lambda x, y: x.view(-1)[y[3].view(-1).data.long()].view(1, -1), labels_train_list,
                    atlasPatch_recons))
        # ========================================================== #

    # ========Loop on test examples======================== #
    print("Start : ", category)
    iterator = trainer.dataloader_test.__iter__()
    with torch.no_grad():
        for i in tqdm.tqdm(range(trainer.len_dataset_test)):
            try:
                P2, _, _, P2_path = trainer.dataset_test[i]
                P2_label = P2[:, 6].data.cpu().numpy()
                P2 = P2[:, :3].unsqueeze(0).contiguous().cuda().float()
                P2_latent = trainer.network.encode(P2.transpose(1, 2).contiguous(), P2.transpose(1, 2).contiguous())
            except:
                break

            # Chamfer (P0_P2)
            P0_P2_list = list(
                map(lambda x: loss.forward_chamfer(trainer.network, P2, x, local_fix=None, distChamfer=distChamfer),
                    points_train_list))


            # Ensemble method
            def emsemble(top_k_idx, predicted_list):
                predicted_emsemble = predicted_list[top_k_idx]
                Ensemble = my_utils.get_emsemble(predicted_emsemble)
                iou = miou_shape.miou_shape(Ensemble, P2_label, trainer.parts)
                return iou


            iou_dict = {}


            def add(name, top, ensemble=True, ensemble_list=None):
                iou_dict[name + "_NN"] = iou_NN_list[top[0]]
                iou_dict[name + "_ours"] = iou_ours_list[top[0]]
                if use_atlas_baseline:
                    iou_dict[name + "_atlasSphere"] = iou_atlasSphere_list[top[0]]
                    iou_dict[name + "_atlasPatch"] = iou_atlasPatch_list[top[0]]

                if ensemble:
                    iou_dict[name + "_ours_emsemble"] = emsemble(top, predicted_ours_P2_P0_list)
                    iou_dict[name + "_NN_emsemble"] = emsemble(top, predicted_NN_P2_P0_list)

                    if use_atlas_baseline:
                        iou_dict[name + "_atlasSphere_emsemble"] = emsemble(top, predicted_atlasSphere_list)
                        iou_dict[name + "_atlasPatch_emsemble"] = emsemble(top, predicted_atlasPatch_list)


            if use_atlas_baseline:
                ## Atlasnet
                P2_atlasSphere, dist1_P2_atlasSphere, dist2_P2_atlasSphere, idx1_P2_atlasSphere, idx2_P2_atlasSphere = loss.forward_chamfer_atlasnet(
                    network_atlasSphere, P2, distChamfer=distChamfer)
                predicted_atlasSphere_list = list(
                    map(lambda x: x.view(-1)[idx2_P2_atlasSphere.view(-1).data.long()].view(1, -1), atlasSphere_labels))
                iou_atlasSphere_list = list(
                    map(lambda x: miou_shape.miou_shape(x.squeeze().cpu().numpy(), P2_label, trainer.parts),
                        predicted_atlasSphere_list))
                predicted_atlasSphere_list = torch.cat(predicted_atlasSphere_list)

                P2_atlasPatch, dist1_P2_atlasPatch, dist2_P2_atlasPatch, idx1_P2_atlasPatch, idx2_P2_atlasPatch = loss.forward_chamfer_atlasnet(
                    network_atlasPatch, P2, distChamfer=distChamfer)
                predicted_atlasPatch_list = list(
                    map(lambda x: x.view(-1)[idx2_P2_atlasPatch.view(-1).data.long()].view(1, -1), atlasPatch_labels))
                iou_atlasPatch_list = list(
                    map(lambda x: miou_shape.miou_shape(x.squeeze().cpu().numpy(), P2_label, trainer.parts),
                        predicted_atlasPatch_list))
                predicted_atlasPatch_list = torch.cat(predicted_atlasPatch_list)

            # Compute Chamfer (P2_P0)
            P2_P0_list = list(
                map(lambda x: loss.forward_chamfer(trainer.network, x, P2, local_fix=None, distChamfer=distChamfer),
                    points_train_list))
            predicted_ours_P2_P0_list = list(
                map(lambda x, y: x.view(-1)[y[4].view(-1).data.long()].view(1, -1), labels_train_list, P2_P0_list))
            iou_ours_list = list(
                map(lambda x: miou_shape.miou_shape(x.squeeze().cpu().numpy(), P2_label, trainer.parts),
                    predicted_ours_P2_P0_list))
            predicted_ours_P2_P0_list = torch.cat(predicted_ours_P2_P0_list)

            # Compute NN
            P2_P0_NN_list = list(map(lambda x: distChamfer(x, P2), points_train_list))
            predicted_NN_P2_P0_list = list(
                map(lambda x, y: x.view(-1)[y[3].view(-1).data.long()].view(1, -1), labels_train_list, P2_P0_NN_list))
            iou_NN_list = list(map(lambda x: miou_shape.miou_shape(x.squeeze().cpu().numpy(), P2_label, trainer.parts),
                                   predicted_NN_P2_P0_list))
            predicted_NN_P2_P0_list = torch.cat(predicted_NN_P2_P0_list)

            # NN
            NN_chamferL2_list = list(map(lambda x: loss.chamferL2(x[0], x[1]), P2_P0_NN_list))
            top_k_idx, top_k_values = min_k(NN_chamferL2_list)
            add("iou_NN", top_k_idx)

            # NN + ICP
            points_train_NN_ICP = ICP.ICP(points_train_list[top_k_idx[0]], P2).unsqueeze(0).float()
            dist1_NN_tr, dist2_NN_tr, idx1_NN_tr, idx2_NN_tr = distChamfer(points_train_NN_ICP, P2)
            predicted_P2_NN_tr = labels_train_list[top_k_idx[0]].view(-1)[idx2_NN_tr.view(-1).data.long()].view(-1)
            iou_dict["iou_NN_ICP_NN"] = miou_shape.miou_shape(predicted_P2_NN_tr.cpu().numpy(), P2_label, trainer.parts)

            # NN + ICP + ours
            P2_P1_ours_1, _, _, _, idx2_P2_P0 = loss.forward_chamfer(trainer.network, points_train_NN_ICP, P2,
                                                                     local_fix=None, distChamfer=distChamfer)
            iou_dict["iou_NN_ICP_ours"] = miou_shape.miou_shape(
                labels_train_list[top_k_idx[0]].view(-1)[idx2_P2_P0.view(-1).data.long()].view(1,
                                                                                               -1).squeeze().cpu().numpy(),
                P2_label,
                trainer.parts)

            # Chamfer ours
            chamfer_list = list(map(lambda x: loss.chamferL2(x[1], x[2]), P2_P0_list))
            top_k_idx, top_k_values = min_k(chamfer_list)
            add("iou_chamfer", top_k_idx)

            # NN in latent space
            P0_latent_list = list(
                map(lambda x: trainer.network.encode(x.transpose(1, 2).contiguous(),
                                                     x.transpose(1, 2).contiguous()),
                    points_train_list))
            cosine_list = list(map(lambda x: loss.cosine(x, P2_latent), P0_latent_list))

            top_k_idx, top_k_values = min_k(cosine_list)
            add("iou_cosine", top_k_idx)

            # Cycle 2
            P0_P2_cycle_list = list(map(lambda x, y: loss.batch_cycle_2(x[0], y[3], 1), P0_P2_list, P2_P0_list))
            P0_P2_cycle_list = list(map(lambda x, y: loss.L2(x, y), P0_P2_cycle_list, points_train_list))
            top_k_idx, top_k_values = min_k(P0_P2_cycle_list)
            iou_P0_P2_cycle_ours = iou_ours_list[top_k_idx[0]]
            iou_P0_P2_cycle_NN = iou_NN_list[top_k_idx[0]]

            P2_P0_cycle_list = list(map(lambda x, y: loss.batch_cycle_2(x[0], y[3], 1), P2_P0_list, P0_P2_list))
            P2_P0_cycle_list = list(map(lambda x: loss.L2(x, P2), P2_P0_cycle_list))
            top_k_idx, top_k_values = min_k(P2_P0_cycle_list)
            iou_P2_P0_cycle_ours = iou_ours_list[top_k_idx[0]]
            iou_P2_P0_cycle_NN = iou_NN_list[top_k_idx[0]]

            # Cycle 2 both sides
            both_cycle_list = list(map(lambda x, y: x * y, P0_P2_cycle_list, P2_P0_cycle_list))
            both_cycle_list = np.power(both_cycle_list, 1.0 / 2.0).tolist()
            top_k_cycle2_idx, top_k_values = min_k(both_cycle_list)
            add("both_cycle", top_k_cycle2_idx)

            # ORACLE OURS
            top_k_idx, top_k_values = max_k(iou_ours_list)
            add("oracle_ours", top_k_idx, ensemble=False)

            # ORACLE NN
            top_k_idx, top_k_values = max_k(iou_NN_list)
            add("oracle_NN", top_k_idx, ensemble=False)
            if opt.num_figure_3_4 > 0 and i < 50 : #Put this in a if statement or causes memory leak
                top_cycles.append((top_k_cycle2_idx[0], top_k_values[0], iou_ours_list[
                    top_k_cycle2_idx[0]], iou_NN_list[top_k_cycle2_idx[0]], predicted_NN_P2_P0_list[top_k_cycle2_idx[0]],
                                   predicted_ours_P2_P0_list[top_k_cycle2_idx[0]],
                                   P2_P0_list[top_k_cycle2_idx[0]], P2.cpu(), P2_label, P2_path))

            list(map(update_meter, local_meter, criterions))

        # Add to global str
        list(map(get_update_global_meter(trainer.len_dataset_test), global_meter, local_meter))
        global_str = list(map(print_avg, global_str, local_meter, criterions))

    if opt.num_figure_3_4 > 0:
        makefigure = figure_2_3.MakeFigure(opt, top_cycles, max_k_fig, min_k_fig, category, path_train_list, points_train_list,
                                   labels_train_list, trainer, trainer.parts, distChamfer)
        makefigure.make_figure_2_3()

global_str = list(map(print_global_avg, global_str, global_meter, criterions))

# Final results
global_str = list(map(lambda x: x + " \\\\" + '\n', global_str))
with open("table.txt", 'a') as fp:
    fp.write(" " + '\n')
    fp.write(opt.logdir + " " + str(opt.num_shots_eval) + '\n')
    fp.write(reduce(lambda x, y: x + y, global_str))
# print(reduce(lambda x, y: x + y, global_str))


with open('ours' + opt.categories[0] + '.txt', 'a') as f:  # open and append
    f.write(" " + '\n')
    f.write(reduce(lambda x, y: x + y, global_str))
print(global_str)
print(global_meter)

for index, key in enumerate(criterions):
    trainer.local_dict_to_save_experiment[key] = global_meter[index].avg

trainer.save_new_experiments_results()

