import pymesh
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(BASE_DIR + "/../")
import normalize_points
import torch

def link(path1):
	"""
	This function takes a path to the orginal shapenet model and subsample it nicely
	"""
	obj1 = pymesh.load_mesh(path1)
	if len(obj1.vertices)<10000:
		obj1 = pymesh.split_long_edges(obj1, 0.02)[0]
		while len(obj1.vertices)<10000:
			obj1 = pymesh.subdivide(obj1)

	new_mesh = pymesh.form_mesh(normalize_points.BoundingBox(torch.from_numpy(obj1.vertices)).numpy(), obj1.faces)
	return new_mesh


if __name__ == '__main__':
	main()