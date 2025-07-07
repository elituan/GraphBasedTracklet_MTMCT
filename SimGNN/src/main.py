"""SimGNN runner for trajectory similarity."""

import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import glob
from utils import tab_printer
from simgnn import SimGNNTrainer
from param_parser import parameter_parser

def extract_trajectory_info(track_data):
    """
    Extract trajectory information from track data.
    
    Args:
        track_data: Dictionary containing track information
        
    Returns:
        sequence_name: Name of the video sequence
        track_id: ID of the track
        frames: List of frame IDs in order
        features: List of features corresponding to frames
        bboxes: List of bounding boxes corresponding to frames
    """
    # Get all image names and sort them by frame number
    image_names = list(track_data.keys())
    # Extract sequence name and track ID from the first image name
    parts = image_names[0].split('_')
    sequence_name = parts[0]
    track_id = parts[1]
    
    # Sort images by frame number
    image_names.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    
    frames = []
    features = []
    bboxes = []
    
    for img_name in image_names:
        img_data = track_data[img_name]
        frames.append(img_data['frame'])
        features.append(img_data['feat'])
        bboxes.append(img_data['bbox'])
    
    return sequence_name, track_id, frames, features, bboxes

def build_trajectory_graph(features, frames, threshold=0.5):
    """
    Build a graph representation of a trajectory.
    
    Args:
        features: List of feature vectors
        frames: List of frame IDs
        threshold: Distance threshold for connecting nodes (default: 0.5)
        
    Returns:
        graph_data: Dictionary with graph structure (edges) and node features
    """
    # Initialize the graph data structure
    graph_data = {
        "graph": [],
        "labels": features
    }
    
    # Build edges based on Euclidean distance between features
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            # Calculate Euclidean distance between feature vectors
            feat_i = np.array(features[i])
            feat_j = np.array(features[j])
            distance = np.linalg.norm(feat_i - feat_j)
            
            # Add edge if distance is below threshold
            if distance < threshold:
                graph_data["graph"].append([i, j])
    
    # If no edges were created (all distances > threshold), 
    # at least connect sequential nodes to ensure a connected graph
    if len(graph_data["graph"]) == 0:
        for i in range(len(features)-1):
            graph_data["graph"].append([i, i+1])
    
    return graph_data

def graph_to_data_dict(G):
    """
    Convert graph data dictionary to data dictionary format for SimGNN.
    
    Args:
        G: Graph data dictionary with "graph" and "labels" keys
        
    Returns:
        data: Dictionary with graph structure and node features
    """
    # The graph is already in the correct format, just return it
    data = {
        "graph": G["graph"],
        "features": G["labels"],
        # "labels": list(range(len(G["labels"])))  # Use node indices as labels
        
        "labels": [2] * len(G["labels"])  # Use default label 2 for all nodes
    }
    
    return data

def compute_graph_pair_simi(trainer, graph1, graph2):
        """
        Compute similarity between a pair of trajectory graphs.
        
        Args:
            trainer: SimGNN trainer object
            graph1: First trajectory graph
            graph2: Second trajectory graph
            
        Returns:
            similarity: Similarity score between the two graphs
        """
        # Convert graphs to data dictionary format
        data1 = graph_to_data_dict(graph1)
        data2 = graph_to_data_dict(graph2)
        # print(f"data1: {data1}")
        # print(f"data2: {data2}")
    
        
        # Prepare data for SimGNN
        sim_data = {}
        sim_data["graph_1"] = data1["graph"]
        sim_data["graph_2"] = data2["graph"]
        sim_data["labels_1"] = data1["labels"]
        sim_data["labels_2"] = data2["labels"]
        
        # Estimate graph edit distance
        sim_data["ged"] = max(len(data1["labels"]), len(data2["labels"]))
        # print(f"sim_data: {sim_data}")


        # add later
        # self.ground_truth.append(calculate_normalized_ged(data))

        # Convert to torch tensors
        torch_data = trainer.transfer_to_torch(sim_data)
        # print(f"torch_data: {torch_data}")
        
        # Get similarity prediction
        trainer.model.eval()
        with torch.no_grad():
            prediction = trainer.model(torch_data).item()
        
        return prediction

def compute_graph_similarities(trajectory_graphs, trainer):
    """
    Compute similarities between all pairs of trajectory graphs.
    
    Args:
        trajectory_graphs: Dictionary of trajectory graphs
        trainer: SimGNN trainer object
        
    Returns:
        similarities: Dictionary of similarities between trajectories
    """
    similarities = {}
    track_ids = list(trajectory_graphs.keys())
    
    trainer.model.eval()
    
    for i, track_id1 in enumerate(tqdm(track_ids, desc="Computing similarities")):
        similarities[track_id1] = {}
        G1 = trajectory_graphs[track_id1]
        
        for j, track_id2 in enumerate(track_ids):
            if i == j:  # Same graph
                similarities[track_id1][track_id2] = 1.0
                continue
                
            G2 = trajectory_graphs[track_id2]
            similarities[track_id1][track_id2] = compute_graph_pair_simi(trainer, G1, G2)
    
    return similarities

def process_single_pkl(pkl_path, trainer):
    """
    Process a single pickle file.
    
    Args:
        pkl_path: Path to the pickle file
        trainer: SimGNN trainer object
        
    Returns:
        trajectory_graphs: Dictionary of trajectory graphs
    """
    print(f"Processing: {pkl_path}")
    
    # Load trajectory data from pickle file
    with open(pkl_path, 'rb') as f:
        trajectory_data = pickle.load(f)
    
    print(f"  Number of detections: {len(trajectory_data)}")
    
    # Group trajectories by track ID
    tracks = {}
    for img_name, img_data in trajectory_data.items():
        parts = img_name.split('_')
        if len(parts) >= 3:
            track_id = parts[1]
            if track_id not in tracks:
                tracks[track_id] = {}
            tracks[track_id][img_name] = img_data
    
    # Build graphs for each trajectory
    print(" Building trajectory graphs")
    trajectory_graphs = {}
    for track_id, track_data in tqdm(tracks.items(), desc=" Building graphs"):
        sequence_name, track_id, frames, features, bboxes = extract_trajectory_info(track_data)
        # Only create graphs for trajectories with enough frames
        if len(frames) >= 5:
            trajectory_graphs[track_id] = build_trajectory_graph(features, frames)
    # return trajectory_graphs

    
    # Compute similarities between all trajectory graphs
    print("  Computing similarities between trajectories")
    similarities = compute_graph_similarities(trajectory_graphs, trainer)
    
    # Add similarities to the original data
    for track_id, track_similarities in similarities.items():
        for img_name in tracks[track_id]:
            trajectory_data[img_name]['similarities'] = track_similarities
    
    # Save the updated trajectory data back to the pickle file
    output_path = pkl_path
    print(f"  Saving results to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(trajectory_data, f)

def main():
    """
    Load trajectories, build graphs, compute similarities, and save results.
    """
    args = parameter_parser()
    tab_printer(args)
    
    # Check if pkl_path is provided
    if not args.pkl_path:
        print("Please provide a path to the trajectory pickle file or directory using --pkl-path")
        return
    
    # Initialize SimGNN trainer
    trainer = SimGNNTrainer(args)
    
    # Load pre-trained model if provided
    if args.load_path:
        print(f"Loading pre-trained model from {args.load_path}")
        trainer.load()
    else:
        print("Training new SimGNN model")
        trainer.fit()
    
    # Check if the path is a directory
    if os.path.isdir(args.pkl_path):
        # Find all matching pkl files in subdirectories (c041, c042, ...)
        pkl_pattern = os.path.join(args.pkl_path, 'c0??', 'c0??_mot_feat.pkl')
        pkl_files = glob.glob(pkl_pattern)
        cam_ids = [pkl_file.split('/')[-2] for pkl_file in pkl_files]
        
        if not pkl_files:
            print(f"No matching pickle files found in {args.pkl_path}")
            return
        
        print(f"Found {len(pkl_files)} pickle files to process")
        
        # Process each pickle file
        trajectory_graph_all = {}
        for i, pkl_file in enumerate(pkl_files):
            trajectory_graphs = process_single_pkl(pkl_file, trainer)
            trajectory_graph_all[cam_ids[i]] = trajectory_graphs
    else:
        # Process a single pickle file
        trajectory_graph_all = process_single_pkl(args.pkl_path, trainer)
    
    # Save trajectory_graph_all to pickle file
    output_pkl_path = os.path.join(os.path.dirname(args.pkl_path), 'trajectory_graph_all.pkl')
    print(f"Saving trajectory_graph_all to {output_pkl_path}")
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(trajectory_graph_all, f, pickle.HIGHEST_PROTOCOL)

    print("All processing complete!")

if __name__ == "__main__":
    main()
