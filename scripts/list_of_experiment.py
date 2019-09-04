class Experiments(object):
    def __init__(self, shapenet_path = None):
        self.shapenet_path = shapenet_path
        self.opts = {
            "sup_1": "--logdir Airplane_sup --cat Airplane --part_supervision 1",
            "sup_2": "--logdir Car_sup --cat Car --part_supervision 1",
            "sup_3": "--logdir Chair_sup --cat Chair --part_supervision 1",
            "sup_4": "--logdir Lamp_sup --cat Lamp --part_supervision 1",
            "sup_5": "--logdir Table_sup --cat Table --part_supervision 1",

            "unsup_1": "--logdir Airplane_unsup --cat Airplane --part_supervision 0",
            "unsup_2": "--logdir Car_unsup --cat Car --part_supervision 0",
            "unsup_3": "--logdir Chair_unsup --cat Chair --part_supervision 0",
            "unsup_4": "--logdir Lamp_unsup --cat Lamp --part_supervision 0",
            "unsup_5": "--logdir Table_unsup --cat Table --part_supervision 0",

            "ablation_1": "--logdir Car_unsup_ablation_1 --cat Car --part_supervision 0 --lambda_cycle_2 0 --lambda_cycle_3 0 --num_shots_eval 100",
            "ablation_2": "--logdir Car_unsup_ablation_2 --cat Car --part_supervision 0 --lambda_chamfer 0 --num_shots_eval 100",
            "ablation_3": "--logdir Car_unsup_ablation_3 --cat Car --part_supervision 0 --knn 0 --num_shots_eval 100",

            "atlasnet_sphere_1": "--train_atlasnet 1 --logdir Atlasnet_Sphere_Airplane --knn 0 --cat Airplane --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Airplane/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_sphere_2": "--train_atlasnet 1 --logdir Atlasnet_Sphere_Car --knn 0 --cat Car --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Car/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_sphere_3": "--train_atlasnet 1 --logdir Atlasnet_Sphere_Chair --knn 0 --cat Chair --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Chair/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_sphere_4": "--train_atlasnet 1 --logdir Atlasnet_Sphere_Lamp --knn 0 --cat Lamp --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Lamp/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_sphere_5": "--train_atlasnet 1 --logdir Atlasnet_Sphere_Table --knn 0 --cat Table --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Table/network.pth --randomize 1 --num_shots_eval 10",

            "atlasnet_patch_1": "--train_atlasnet 1 --logdir Atlasnet_Patch_Airplane --knn 0 --cat Airplane --atlasnet PATCH --atlasPatch_path log/Atlasnet_Patch_Airplane/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_patch_2": "--train_atlasnet 1 --logdir Atlasnet_Patch_Car --knn 0 --cat Car --atlasnet PATCH --atlasPatch_path log/Atlasnet_Patch_Car/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_patch_3": "--train_atlasnet 1 --logdir Atlasnet_Patch_Chair --knn 0 --cat Chair --atlasnet PATCH --atlasPatch_path log/Atlasnet_Patch_Chair/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_patch_4": "--train_atlasnet 1 --logdir Atlasnet_Patch_Lamp --knn 0 --cat Lamp --atlasnet PATCH --atlasPatch_path log/Atlasnet_Patch_Lamp/network.pth --randomize 1 --num_shots_eval 10",
            "atlasnet_patch_5": "--train_atlasnet 1 --logdir Atlasnet_Patch_Table --knn 0 --cat Table --atlasnet PATCH --atlasPatch_path log/Atlasnet_Patch_Table/network.pth --randomize 1 --num_shots_eval 10",
        }


        self.trainings = {}
        for key in self.opts:
            self.trainings[key] = "python training/train_shapenet.py " + self.opts[key]


        self.inference_table_2_3 = {
            "sup_1": self.opts["sup_1"],
            "sup_2": self.opts["sup_2"],
            "sup_3": self.opts["sup_3"],
            "sup_4": self.opts["sup_4"],
            "sup_5": self.opts["sup_5"],

            "unsup_1": self.opts["unsup_1"] + " --num_figure_3_4 3  --shapenetv1_path " + str(self.shapenet_path),
            "unsup_2": self.opts["unsup_2"] + " --num_figure_3_4 3  --shapenetv1_path " + str(self.shapenet_path),
            "unsup_3": self.opts["unsup_3"] + " --num_figure_3_4 3  --shapenetv1_path " + str(self.shapenet_path),
            "unsup_4": self.opts["unsup_4"] + " --num_figure_3_4 3  --shapenetv1_path " + str(self.shapenet_path),
            "unsup_5": self.opts["unsup_5"] + " --num_figure_3_4 3  --shapenetv1_path " + str(self.shapenet_path),

            "ablation_0": "--logdir Car_unsup --cat Car --part_supervision 0 --num_shots_eval 100",
            "ablation_1": self.opts["ablation_1"],
            "ablation_2": self.opts["ablation_2"],
            "ablation_3": self.opts["ablation_3"],
        }
        for key in self.inference_table_2_3:
            self.inference_table_2_3[key] = "python inference/eval_segmentation.py " + self.inference_table_2_3[key]

        self.inference_table_1 = {
            "atlasnet_sphere_1": " --logdir Airplane_unsup --knn 0 --cat Airplane --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Airplane/network.pth --randomize 1 --num_shots_eval 10 --atlasPatch_path log/Atlasnet_Patch_Airplane/network.pth",
            "atlasnet_sphere_2": " --logdir Car_unsup --knn 0 --cat Car --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Car/network.pth --randomize 1 --num_shots_eval 10 --atlasPatch_path log/Atlasnet_Patch_Car/network.pth",
            "atlasnet_sphere_3": " --logdir Chair_unsup --knn 0 --cat Chair --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Chair/network.pth --randomize 1 --num_shots_eval 10 --atlasPatch_path log/Atlasnet_Patch_Chair/network.pth",
            "atlasnet_sphere_4": " --logdir Lamp_unsup --knn 0 --cat Lamp --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Lamp/network.pth --randomize 1 --num_shots_eval 10 --atlasPatch_path log/Atlasnet_Patch_Lamp/network.pth",
            "atlasnet_sphere_5": " --logdir Table_unsup --knn 0 --cat Table --atlasnet SPHERE --atlasSphere_path log/Atlasnet_Sphere_Table/network.pth --randomize 1 --num_shots_eval 10 --atlasPatch_path log/Atlasnet_Patch_Table/network.pth",
        }
        print("Inferences in inference_table_1 are to be run 10 times and averaged")
        for key in self.inference_table_1:
            self.inference_table_1[key] = "python inference/eval_segmentation.py " + self.inference_table_1[key]

        # Additional Stuff : to remove
        self.additional_trainings = {
            "new_tr_ablation_1": "python training/train_shapenet.py --logdir Chair_unsup_ablation_1 --cat Chair --part_supervision 0 --lambda_cycle_2 0 --lambda_cycle_3 0 --num_shots_eval 5000",
            "new_tr_ablation_2": "python training/train_shapenet.py --logdir Chair_unsup_ablation_2 --cat Chair --part_supervision 0 --lambda_chamfer 0 --num_shots_eval 5000",
            "new_tr_ablation_3": "python training/train_shapenet.py --logdir Chair_unsup_ablation_3 --cat Chair --part_supervision 0 --knn 0 --num_shots_eval 5000",
            "new_tr_ablation_1": "python training/train_shapenet.py --logdir Airplane_unsup_ablation_1 --cat Airplane --part_supervision 0 --lambda_cycle_2 0 --lambda_cycle_3 0 --num_shots_eval 5000",
            "new_tr_ablation_2": "python training/train_shapenet.py --logdir Airplane_unsup_ablation_2 --cat Airplane --part_supervision 0 --lambda_chamfer 0 --num_shots_eval 5000",
            "new_tr_ablation_3": "python training/train_shapenet.py --logdir Airplane_unsup_ablation_3 --cat Airplane --part_supervision 0 --knn 0 --num_shots_eval 5000",

        }

        self.additional_inference = {
            "new_ablation_0": "python inference/eval_segmentation.py  --logdir Car_unsup --cat Car --part_supervision 0 --num_shots_eval 5000",
            "new_ablation_1": "python inference/eval_segmentation.py  --logdir Car_unsup_ablation_1 --cat Car --part_supervision 0 --lambda_cycle_2 0 --lambda_cycle_3 0 --num_shots_eval 5000",
            "new_ablation_2": "python inference/eval_segmentation.py  --logdir Car_unsup_ablation_2 --cat Car --part_supervision 0 --lambda_chamfer 0 --num_shots_eval 5000",
            "new_ablation_3": "python inference/eval_segmentation.py  --logdir Car_unsup_ablation_3 --cat Car --part_supervision 0 --knn 0 --num_shots_eval 5000",
            "new_ablation_0": "python inference/eval_segmentation.py  --logdir Chair_unsup --cat Chair --part_supervision 0 --num_shots_eval 5000",
            "new_ablation_1": "python inference/eval_segmentation.py  --logdir Chair_unsup_ablation_1 --cat Chair --part_supervision 0 --lambda_cycle_2 0 --lambda_cycle_3 0 --num_shots_eval 5000",
            "new_ablation_2": "python inference/eval_segmentation.py  --logdir Chair_unsup_ablation_2 --cat Chair --part_supervision 0 --lambda_chamfer 0 --num_shots_eval 5000",
            "new_ablation_3": "python inference/eval_segmentation.py  --logdir Chair_unsup_ablation_3 --cat Chair --part_supervision 0 --knn 0 --num_shots_eval 5000",
            "new_ablation_0": "python inference/eval_segmentation.py  --logdir Airplane_unsup --cat Airplane --part_supervision 0 --num_shots_eval 5000",
            "new_ablation_1": "python inference/eval_segmentation.py  --logdir Airplane_unsup_ablation_1 --cat Airplane --part_supervision 0 --lambda_cycle_2 0 --lambda_cycle_3 0 --num_shots_eval 5000",
            "new_ablation_2": "python inference/eval_segmentation.py  --logdir Airplane_unsup_ablation_2 --cat Airplane --part_supervision 0 --lambda_chamfer 0 --num_shots_eval 5000",
            "new_ablation_3": "python inference/eval_segmentation.py  --logdir Airplane_unsup_ablation_3 --cat Airplane --part_supervision 0 --knn 0 --num_shots_eval 5000",
        }

if __name__ == '__main__':
    exp = Experiments()
    print(exp.inference_table_2_3)
    print(exp.inference_table_1)
    print(exp.trainings)
