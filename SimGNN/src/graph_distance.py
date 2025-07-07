#!/usr/bin/env python
"""
Graph distance calculation utility for trajectory graphs.
"""

import os
import pickle
import argparse
import numpy as np
import torch
import time
from tqdm import tqdm
from simgnn import SimGNNTrainer
from param_parser import parameter_parser
from main import compute_graph_pair_simi
from utils import tab_printer
import random

def load_trajectory_graphs(pkl_path):
    """
    Load trajectory graphs from pickle file.
    
    Args:
        pkl_path: Path to the pickle file containing trajectory graphs
        
    Returns:
        trajectory_graphs: Dictionary of trajectory graphs
    """
    print(f"Loading trajectory graphs from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        trajectory_graphs = pickle.load(f)
    
    # Print summary information
    if isinstance(trajectory_graphs, dict):
        if all(isinstance(v, dict) for v in trajectory_graphs.values()):
            # Multi-camera structure
            print("Multi-camera trajectory graphs loaded:")
            for cam_id, graphs in trajectory_graphs.items():
                print(f"  Camera {cam_id}: {len(graphs)} trajectories")
        else:
            # Single camera structure
            print(f"Single camera trajectory graphs loaded: {len(trajectory_graphs)} trajectories")
    
    return trajectory_graphs

def calculate_cross_camera_distances(trajectory_graphs, trainer, source_cam, target_cam, threshold=None):
    """
    Calculate distances between trajectories from different cameras.
    
    Args:
        trajectory_graphs: Dictionary of trajectory graphs by camera
        trainer: SimGNN trainer object
        source_cam: Source camera ID
        target_cam: Target camera ID
        threshold: Optional similarity threshold to filter results
        
    Returns:
        distances: Dictionary of distances between trajectories
    """
    if source_cam not in trajectory_graphs or target_cam not in trajectory_graphs:
        print(f"Error: Camera {source_cam} or {target_cam} not found in trajectory graphs")
        return {}
    
    source_graphs = trajectory_graphs[source_cam]
    target_graphs = trajectory_graphs[target_cam]
    
    print(f"Calculating distances between camera {source_cam} ({len(source_graphs)} trajectories) "
          f"and camera {target_cam} ({len(target_graphs)} trajectories)")
    
    distances = {}
    for source_id, source_graph in tqdm(source_graphs.items(), desc=f"{source_cam}->{target_cam}"):
        distances[source_id] = {}
        for target_id, target_graph in target_graphs.items():
            similarity = compute_graph_pair_simi(trainer, source_graph, target_graph)
            # Convert similarity to distance (1 - similarity)
            distance = 1.0 - similarity
            
            # Only include distances below threshold if specified
            if threshold is None or distance < threshold:
                distances[source_id][target_id] = distance
    
    return distances

def find_closest_trajectories(distances, top_k=5):
    """
    Find the closest trajectories for each source trajectory.
    
    Args:
        distances: Dictionary of distances between trajectories
        top_k: Number of closest trajectories to return
        
    Returns:
        closest: Dictionary of closest trajectories
    """
    closest = {}
    for source_id, target_distances in distances.items():
        # Sort targets by distance (ascending)
        sorted_targets = sorted(target_distances.items(), key=lambda x: x[1])
        # Take top k
        closest[source_id] = sorted_targets[:top_k]
    
    return closest

def main():
    """
    Main function to load trajectory graphs and calculate distances.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate distances between trajectory graphs")
    parser.add_argument("--pkl-path", type=str, required=True, 
                        help="Path to the trajectory_graph_all.pkl file")
    # parser.add_argument("--source-cam", type=str, required=True,
    #                     help="Source camera ID (e.g., c041)")
    # parser.add_argument("--target-cam", type=str, required=True,
    #                     help="Target camera ID (e.g., c042)")
    # parser.add_argument("--threshold", type=float, default=None,
    #                     help="Distance threshold to filter results (optional)")
    # parser.add_argument("--top-k", type=int, default=5,
    #                     help="Number of closest trajectories to return (default: 5)")
    # parser.add_argument("--output", type=str, default=None,
    #                     help="Path to save the output pickle file (optional)")
    parser = argparse.ArgumentParser(description="Run SimGNN.")

    parser.add_argument("--training-graphs",
                        nargs="?",
                        default="../dataset/train/",
	                help="Folder with training graph pair jsons.")

    parser.add_argument("--testing-graphs",
                        nargs="?",
                        default="../dataset/test/",
	                help="Folder with testing graph pair jsons.")

    parser.add_argument("--pkl-path",
                        type=str,
                        default="",
                        help="Path to the trajectory pickle file.")

    parser.add_argument("--epochs",
                        type=int,
                        default=5,
	                help="Number of training epochs. Default is 5.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=128,
	                help="Filters (neurons) in 1st convolution. Default is 128.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=64,
	                help="Filters (neurons) in 2nd convolution. Default is 64.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=32,
	                help="Filters (neurons) in 3rd convolution. Default is 32.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
	                help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
	                help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
	                help="Similarity score bins. Default is 16.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        action="store_true")

    parser.set_defaults(histogram=False)

    parser.add_argument("--save-path",
                        type=str,
                        default="",
                        help="Where to save the trained model")

    parser.add_argument("--load-path",
                        type=str,
                        default="",
                        help="Load a pretrained model from this path")
    
    cmd_args = parser.parse_args()
    tab_printer(cmd_args)

    # Initialize SimGNN trainer with default parameters
    # Use a separate namespace for SimGNN parameters to avoid conflicts
    # cmd_args = parameter_parser([])  # Pass empty list to avoid using sys.argv
    trainer = SimGNNTrainer(cmd_args)
    
    # Load pre-trained model
    # print(f"Loading pre-trained model from {cmd_args.model_path}")
    trainer.load()
    
    # Load trajectory graphs
    start_loading_trajectory_graphs = time.time()
    trajectory_graphs = load_trajectory_graphs(cmd_args.pkl_path)
    end_loading_trajectory_graphs = time.time()
    loading_trajectory_graphs_time = end_loading_trajectory_graphs - start_loading_trajectory_graphs
    print(f"Total loading trajectory graph time: {loading_trajectory_graphs_time:.6f}s")
    
    
    random_key_1 = random.choice(list(trajectory_graphs['c041'].keys()))
    random_key_2 = random.choice(list(trajectory_graphs['c046'].keys()))
    graph1 = trajectory_graphs['c041'][random_key_1]
    graph2 = trajectory_graphs['c046'][random_key_2]


    # Calculate similarity between graph1 and graph2
    start_compute_graph_pair_simi = time.time()
    similarity = compute_graph_pair_simi(trainer, graph1, graph2)
    end_compute_graph_pair_simi = time.time()
    print(f"Similarity between graph1 and graph2: {similarity}")
    compute_graph_pair_simi_time = end_compute_graph_pair_simi - start_compute_graph_pair_simi
    print(f"Total computing graph pair similarity time: {compute_graph_pair_simi_time:.6f}s")
    
    # # Calculate distances between trajectories from different cameras
    # distances = calculate_cross_camera_distances(
    #     trajectory_graphs, trainer, cmd_args.source_cam, cmd_args.target_cam, cmd_args.threshold
    # )
    
    # # Find closest trajectories
    # closest = find_closest_trajectories(distances, cmd_args.top_k)
    
    # # Print results
    # print("\nClosest trajectories:")
    # for source_id, targets in closest.items():
    #     print(f"Source {cmd_args.source_cam}:{source_id} -> ", end="")
    #     for target_id, distance in targets:
    #         print(f"{cmd_args.target_cam}:{target_id} (dist={distance:.4f}), ", end="")
    #     print()
    
    # # Save results if output path is specified
    # if cmd_args.output:
    #     output_data = {
    #         "distances": distances,
    #         "closest": closest,
    #         "source_cam": cmd_args.source_cam,
    #         "target_cam": cmd_args.target_cam
    #     }
    #     with open(cmd_args.output, 'wb') as f:
    #         pickle.dump(output_data, f)
    #     print(f"\nResults saved to {cmd_args.output}")

if __name__ == "__main__":
    main() 