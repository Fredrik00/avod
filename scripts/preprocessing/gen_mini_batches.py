import os

import numpy as np

import avod
from avod.builders.dataset_builder import DatasetBuilder


def do_preprocessing(dataset, indices):

    mini_batch_utils = dataset.kitti_utils.mini_batch_utils

    print("Generating mini batches in {}".format(
        mini_batch_utils.mini_batch_dir))

    # Generate all mini-batches, this can take a long time
    mini_batch_utils.preprocess_rpn_mini_batches(indices)

    print("Mini batches generated")


def split_indices(dataset, num_children):
    """Splits indices between children

    Args:
        dataset: Dataset object
        num_children: Number of children to split samples between

    Returns:
        indices_split: A list of evenly split indices
    """

    all_indices = np.arange(dataset.num_samples)

    # Pad indices to divide evenly
    length_padding = (-len(all_indices)) % num_children
    padded_indices = np.concatenate((all_indices,
                                     np.zeros(length_padding,
                                              dtype=np.int32)))

    # Split and trim last set of indices to original length
    indices_split = np.split(padded_indices, num_children)
    indices_split[-1] = np.trim_zeros(indices_split[-1])

    return indices_split


def split_work(all_child_pids, dataset, indices_split, num_children):
    """Spawns children to do work

    Args:
        all_child_pids: List of child pids are appended here, the parent
            process should use this list to wait for all children to finish
        dataset: Dataset object
        indices_split: List of indices to split between children
        num_children: Number of children
    """

    for child_idx in range(num_children):
        new_pid = os.fork()
        if new_pid:
            all_child_pids.append(new_pid)
        else:
            indices = indices_split[child_idx]
            print('child', dataset.classes,
                  indices_split[child_idx][0],
                  indices_split[child_idx][-1])
            do_preprocessing(dataset, indices)
            os._exit(0)


def main(dataset=None):
    """Generates anchors info which is used for mini batch sampling.

    Processing on 'Cars' can be split into multiple processes, see the Options
    section for configuration.

    Args:
        dataset: KittiDataset (optional)
            If dataset is provided, only generate info for that dataset.
            If no dataset provided, generates info for all 3 classes.
    """

    if dataset is not None:
        do_preprocessing(dataset, None)
        return

    car_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/cars.config'
    ped_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/pedestrians_max_density.config'
    cyc_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/cyclists.config'
    ppl_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/people_max_min_density.config'
    all_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/all.config'
    carped_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/carped/carla_carped.config'
    per_dataset_config_path = avod.root_dir() + \
        '/configs/mb_preprocessing/person.config'

    ##############################
    # Options
    ##############################
    # Serial vs parallel processing
    in_parallel = True

    process_car = False   # Cars
    process_ped = False # Pedestrians
    process_cyc = False  # Cyclists
    process_ppl = False   # People (Pedestrians + Cyclists)
    process_all = False # Cars + Pedestrians + Cyclists
    process_carped = True # Cars + Pedestrians
    process_per = False   # Person (Pedestrians + Cyclists joint class)

    # Number of child processes to fork, samples will
    #  be divided evenly amongst the processes (in_parallel must be True)
    num_car_children = 8
    num_ped_children = 8
    num_cyc_children = 8
    num_ppl_children = 8
    num_all_children = 8
    num_carped_children = 8
    num_per_children = 8

    ##############################
    # Dataset setup
    ##############################
    if process_car:
        car_dataset = DatasetBuilder.load_dataset_from_config(
            car_dataset_config_path)
    if process_ped:
        ped_dataset = DatasetBuilder.load_dataset_from_config(
            ped_dataset_config_path)
    if process_cyc:
        cyc_dataset = DatasetBuilder.load_dataset_from_config(
            cyc_dataset_config_path)
    if process_ppl:
        ppl_dataset = DatasetBuilder.load_dataset_from_config(
            ppl_dataset_config_path)
    if process_all:
        all_dataset = DatasetBuilder.load_dataset_from_config(
            all_dataset_config_path)
    if process_carped:
        carped_dataset = DatasetBuilder.load_dataset_from_config(
            carped_dataset_config_path)
    if process_per:
        per_dataset = DatasetBuilder.load_dataset_from_config(
            per_dataset_config_path)

    ##############################
    # Serial Processing
    ##############################
    if not in_parallel:
        if process_car:
            do_preprocessing(car_dataset, None)
        if process_ped:
            do_preprocessing(ped_dataset, None)
        if process_cyc:
            do_preprocessing(cyc_dataset, None)
        if process_ppl:
            do_preprocessing(ppl_dataset, None)
        if process_all:
            do_preprocessing(all_dataset, None)
        if process_carped:
            do_preprocessing(carped_dataset, None)
        if process_per:
            do_preprocessing(per_dataset, None)

        print('All Done (Serial)')

    ##############################
    # Parallel Processing
    ##############################
    else:

        # List of all child pids to wait on
        all_child_pids = []

        # Cars
        if process_car:
            car_indices_split = split_indices(car_dataset, num_car_children)
            split_work(
                all_child_pids,
                car_dataset,
                car_indices_split,
                num_car_children)

        # Pedestrians
        if process_ped:
            ped_indices_split = split_indices(ped_dataset, num_ped_children)
            split_work(
                all_child_pids,
                ped_dataset,
                ped_indices_split,
                num_ped_children)

        # Cyclists
        if process_cyc:
            cyc_indices_split = split_indices(cyc_dataset, num_cyc_children)
            split_work(
                all_child_pids,
                cyc_dataset,
                cyc_indices_split,
                num_cyc_children)

        # People (Pedestrians + Cyclists)
        if process_ppl:
            ppl_indices_split = split_indices(ppl_dataset, num_ppl_children)
            split_work(
                all_child_pids,
                ppl_dataset,
                ppl_indices_split,
                num_ppl_children)

        # All (Cars + Pedestrians + Cyclists)
        if process_all:
            all_indices_split = split_indices(all_dataset, num_all_children)
            split_work(
                all_child_pids,
                all_dataset,
                all_indices_split,
                num_all_children)

        # Carped (Cars + Pedestrians)
        if process_carped:
            carped_indices_split = split_indices(carped_dataset, num_carped_children)
            split_work(
                all_child_pids,
                carped_dataset,
                carped_indices_split,
                num_carped_children)

        # Person (Pedestrians + Cyclists joint class)
        if process_per:
            per_indices_split = split_indices(per_dataset, num_per_children)
            split_work(
                all_child_pids,
                per_dataset,
                per_indices_split,
                num_per_children)

        # Wait to child processes to finish
        print('num children:', len(all_child_pids))
        for i, child_pid in enumerate(all_child_pids):
            os.waitpid(child_pid, 0)

        print('All Done (Parallel)')


if __name__ == '__main__':
    main()
