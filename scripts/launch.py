import argparse
import os
import list_of_experiment
import gpustat
import time

def parser():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--mode', type=str, default="", choices=['training', 'inference', ''])
    opt = parser.parse_args()
    return opt

opt = parser()


exp = list_of_experiment.Experiments("/trainman-mount/trainman-storage-e7719e4d-b36c-4bc0-a3b3-e13a2d53f66d/ShapeNetCore.v1/")

def get_first_available_gpu():
    """
    Check if a gpu is free and returns it
    :return: gpu_id
    """
    query = gpustat.new_query()
    for gpu_id in range(len(query)):
        gpu = query[gpu_id]
        if gpu.memory_used < 20:
            has = os.system("tmux has-session -t " + f"GPU{gpu_id}" + " 2>/dev/null")
            if not int(has)==0:
                return gpu_id
    return -1


def job_scheduler(dict_of_jobs):
    """
    Launch Tmux session each time it finds a free gpu
    :param dict_of_jobs:
    """
    keys = list(dict_of_jobs.keys())
    while len(keys) > 0:
        job_key = keys.pop()
        job = dict_of_jobs[job_key]
        while get_first_available_gpu() < 0:
            print("Waiting to find a GPU for ", job)
            time.sleep(30) # Sleeps for 30 sec
        gpu_id = get_first_available_gpu()
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {job} 2>&1 | tee  log_terminals/{gpu_id}_{job_key}.txt; tmux kill-session -t GPU{gpu_id}"
        CMD = f'tmux new-session -d -s GPU{gpu_id} \; send-keys "{cmd}" Enter'
        print(CMD)
        os.system(CMD)


if opt.mode == "training":
    print("training mode")
    job_scheduler(exp.trainings)
if opt.mode == "inference":
    print("inference mode")
    job_scheduler(exp.inference_table_1)
    job_scheduler(exp.inference_table_2_3)

print(exp.additional_inference)

# job_scheduler(exp.trainings)
# job_scheduler(exp.inference_table_1)
# job_scheduler(exp.inference_table_2_3)
job_scheduler(exp.additional_inference)
