import numpy as np
import open3d
import torch
import copy
import time
import meter

timings_ICP = meter.AverageValueMeter()  # initialize iou for this shape


def ICP(source, target):
    """

    :param source: source point cloud
    :param target:  target point cloud
    :return: source pointcloud registered
    """
    start = time.time()
    use_torch = False
    if isinstance(source, torch.Tensor):
        use_torch = True
        source = source.squeeze().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.squeeze().cpu().numpy()

    pcd_target = open3d.PointCloud()
    pcd_target.points = open3d.Vector3dVector(target)

    pcd_source = open3d.PointCloud()
    pcd_source.points = open3d.Vector3dVector(source)

    pcd_source = copy.deepcopy(pcd_source)
    pcd_target = copy.deepcopy(pcd_target)

    reg_p2p = open3d.registration_icp(pcd_source, pcd_target, 0.1,
                                      criteria=open3d.ICPConvergenceCriteria(max_iteration=30))

    pcd_source.transform(reg_p2p.transformation)
    source = np.asarray(pcd_source.points)

    if use_torch:
        source = torch.from_numpy(source).cuda()

    end = time.time()
    timings_ICP.update(end - start)
    # print("ellapsed time for a ICP forward pass : ", timings_ICP.value())
    return source
