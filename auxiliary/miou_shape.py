import numpy as np
import auxiliary.meter as meter


def miou_shape(prediction, target, parts):
    # prediction : numpy array size n_elements
    # target : numpy array size n_elements
    # parts : list. e.g. [8.0, 9.0, 10.0, 11.0]

    ious = meter.AverageValueMeter()  # initialize iou for this shape

    for part in parts:
        # select a part
        tp = np.sum((prediction == int(part)) * (target == int(part)))  # true positive
        pc = np.sum((prediction == int(part)))  # predicted positive = TP + FP
        gt = np.sum((target == int(part)))  # GT positive
        if pc + gt - tp == 0:
            # print("union of true positive and predicted positive is empty")
            ious.update(1)
        else:
            ious.update(float(tp) / float(pc + gt - tp))  # add value

    return ious.avg
