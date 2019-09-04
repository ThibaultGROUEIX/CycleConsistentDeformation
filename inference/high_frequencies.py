import numpy as np
import pymesh
import sys
import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/../")

import normalize_points
import useful_losses as loss
import color as mycolor



def colorize_mesh(path):
    mesh = pymesh.load_mesh(path)
    colors = colorize(np.copy(mesh.vertices))
    return colors, mesh


def add_color(mesh, colors):
    output_mesh = pymesh.form_mesh(vertices=mesh.vertices, faces=mesh.faces)

    output_mesh.add_attribute("vertex_red")
    output_mesh.add_attribute("vertex_green")
    output_mesh.add_attribute("vertex_blue")
    output_mesh.set_attribute("vertex_red", colors[:, 0])
    output_mesh.set_attribute("vertex_green", colors[:, 1])
    output_mesh.set_attribute("vertex_blue", colors[:, 2])
    return output_mesh


def save_mesh_with_color(mesh, path):
    pymesh.save_mesh(path, mesh, "vertex_red", "vertex_green", "vertex_blue", ascii=True)


def colorize(vertex):
    vertex_normalized = normalize_points.normalize_unitL2ball_pointcloud(np.expand_dims(vertex, 0)) * 5
    vertex_normalized = np.sum(np.floor(vertex_normalized[0]), 1) % 8
    return mycolor.colors[vertex_normalized.astype('int32')].astype('int32')


def colorize2(vertex):
    vertex_normalized = normalize_points.normalize_unitL2ball_pointcloud(np.expand_dims(vertex, 0))[0]
    vertex_normalized = np.floor(((vertex_normalized + 1) / 2) * 250)
    return vertex_normalized.astype('int32')


def main(input):
    color, mesh = colorize_mesh(input)
    mesh_c = add_color(mesh, color)
    save_mesh_with_color(mesh_c, input)


def high_frequency_propagation(source_path, source_deformed_path, target_path):
    """
    Takes 3 meshes. Define a hig frequency on the first mesh. And propagate frequency to target mesh.
    :param source_path:
    :param source_deformed_path:
    :param target_path:
    :return:
    """
    color, mesh_source = colorize_mesh(source_path)
    mesh_source_c = add_color(mesh_source, color)

    mesh_source_deformed = pymesh.load_mesh(source_deformed_path)
    mesh_source_deformed_c = add_color(mesh_source_deformed, color)

    mesh_target = pymesh.load_mesh(target_path)

    dist1, dist2, idx1, idx2 = loss.distChamfer(torch.from_numpy(mesh_target.vertices).cuda().unsqueeze(0).float(),
                                                torch.from_numpy(
                                                    mesh_source_deformed.vertices).cuda().float().unsqueeze(0))
    color_target = color[idx1.cpu().long().view(-1).numpy()]
    mesh_target_c = add_color(mesh_target, color_target)
    save_mesh_with_color(mesh_source_c, source_path)
    save_mesh_with_color(mesh_target_c, target_path)
    save_mesh_with_color(mesh_source_deformed_c, source_deformed_path)


if __name__ == '__main__':
    high_frequency_propagation(sys.argv[1], sys.argv[2], sys.argv[3])
# main(sys.argv[1])
