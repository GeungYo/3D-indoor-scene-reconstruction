# 3D Indoor Reconstruction

A project for reconstructing indoor 3D space from point cloud data, removing non-structural elements, extracting furniture objects, and preparing the scene for interactive rearrangement.

## Overview

This project focuses on reconstructing a clean indoor scene from point cloud data.

The current pipeline includes:

- building a clean room mesh from structural elements such as walls
- filtering furniture/object points inside the reconstructed room
- detecting furniture regions with bounding boxes
- extracting each furniture point cloud as an independent object
- preparing separated objects for manual editing and rearrangement in Blender

The goal is not only indoor reconstruction, but also object-level scene editing, such as moving and re-placing furniture in a reconstructed 3D room.

## Features

- Clean room mesh generation
- Structural / non-structural separation
- Furniture region detection using bounding boxes
- Object-wise point cloud extraction
- Export of separated furniture objects as individual `.ply` files
- Preparation for interactive 3D scene editing in Blender

## Pipeline

### 1. Room reconstruction
Structural components such as walls are used to generate a clean room mesh.

### 2. Furniture filtering
Only the furniture/object point clouds inside the room are preserved.

### 3. Bounding box detection
Detected furniture regions are represented with bounding boxes.

### 4. Object extraction
Each bounding box is used to crop the corresponding point cloud region.
If needed, clustering can be applied inside each box to further separate merged objects.

### 5. Export for editing
Each extracted object is saved as an independent `.ply` file so that it can be treated as a single furniture object in external tools such as Blender.

## Current Output

The current output includes:

- reconstructed room mesh
- filtered furniture point cloud
- detected furniture bounding boxes
- separated furniture object `.ply` files
- object metadata in `.json`

Example output structure:

```bash
room_detected_results/
separated_objects/
├── object_001.ply
├── object_002.ply
├── object_003.ply
└── objects_meta.json
