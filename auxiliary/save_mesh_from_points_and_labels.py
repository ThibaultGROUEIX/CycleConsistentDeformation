import pymesh
import numpy as np
import torch


def save_mesh_from_pointsandlabels(points, labels=None, path=None, parts=None):
    # points numpy npoints, 3
    # labels , npoints
    colors = np.array([
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
    ])

    if isinstance(points, torch.Tensor):
        points = points.squeeze().cpu().numpy()

    mesh = pymesh.form_mesh(vertices=points, faces=np.array([[0, 1, 2]]))
    if not labels is None:
        if isinstance(labels, torch.Tensor):
            labels = labels.squeeze().cpu().numpy()

        if parts is None:
            set_of_labels = set(labels)
            new_labels = labels.copy()
            j = 1
            for label in set_of_labels:
                new_labels[labels == label] = j
                j = j + 1
        else:
            new_labels = labels.copy()
            j = 1
            for part in parts:
                new_labels[labels == part] = j
                j = j + 1

        new_labels = new_labels.astype('int')
        min_label = min(new_labels) - 1
        new_labels = new_labels - min_label
        colors = colors[min_label:]

        mesh.add_attribute("vertex_red")
        mesh.add_attribute("vertex_green")
        mesh.add_attribute("vertex_blue")
        mesh.set_attribute("vertex_red", colors[new_labels - 1, 0])
        mesh.set_attribute("vertex_green", colors[new_labels - 1, 1])
        mesh.set_attribute("vertex_blue", colors[new_labels - 1, 2])

        pymesh.save_mesh(path, mesh, "vertex_red", "vertex_green", "vertex_blue", ascii=True)
    else:
        pymesh.save_mesh(path, mesh, ascii=True)

    return


if __name__ == '__main__':
    points = np.random.random((1000, 3))
    labels = np.random.randint(0, 6, size=1000)
    save_mesh_from_pointsandlabels(points, labels, "/home/thibault/test.ply")
