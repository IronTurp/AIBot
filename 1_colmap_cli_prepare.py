# -*- coding: utf-8 -*-

import os
import subprocess

class COLMAPProcess:
    def __init__(self, colmap_path, workspace_path, images_path):
        self.colmap_path = colmap_path
        self.workspace_path = workspace_path
        self.images_path = images_path
        self.sparse_path = os.path.join(workspace_path, "sparse")
        self.dense_path = os.path.join(workspace_path, "dense")

        self.check_and_create_directory(self.sparse_path)
        self.check_and_create_directory(self.dense_path)

    def check_and_create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created.")
        else:
            print(f"Directory {path} already exists.")

    def run_command(self, command):
        cmd_string = ' '.join(command)
        subprocess.run(cmd_string)

    def run_feature_extraction(self):
        print("Running feature extraction...")
        cmd = [self.colmap_path, "feature_extractor",
               "--database_path", f"{self.workspace_path}/database.db",
               "--image_path", self.images_path]
        self.run_command(cmd)

    def run_exhaustive_matcher(self):
        print("Running exhaustive matcher...")
        cmd = [self.colmap_path, "exhaustive_matcher",
               "--database_path", f"{self.workspace_path}/database.db"]
        self.run_command(cmd)

    def create_sparse_model(self):
        print("Creating sparse model...")
        cmd = [self.colmap_path, "mapper",
               "--database_path", f"{self.workspace_path}/database.db",
               "--image_path", self.images_path,
               "--output_path", self.sparse_path]
        self.run_command(cmd)

    def create_dense_model(self):
        print("Creating dense model...")
        self.run_command([self.colmap_path, "image_undistorter",
                          "--image_path", self.images_path,
                          "--input_path", f"{self.sparse_path}/0",
                          "--output_path", self.dense_path,
                          "--output_type", "COLMAP",
                          "--max_image_size", "2000"])
        self.run_command([self.colmap_path, "patch_match_stereo",
                          "--workspace_path", self.dense_path])
        self.run_command([self.colmap_path, "stereo_fusion",
                          "--workspace_path", self.dense_path,
                          "--input_type", "photometric",
                          "--output_path", f"{self.dense_path}/fused.ply"])

    def run_poisson_surface_reconstruction(self):
        print("Running Poisson surface reconstruction...")
        self.run_command([self.colmap_path, "poisson_mesher",
                          "--input_path", f"{self.dense_path}/fused.ply",
                          "--output_path", f"{self.dense_path}/meshed-poisson.ply"])

    def process(self):
        self.run_feature_extraction()
        self.run_exhaustive_matcher()
        self.create_sparse_model()
        self.create_dense_model()
        self.run_poisson_surface_reconstruction()
        
colmap = COLMAPProcess(path_to_colmap, 
                       path_to_workspace, 
                       path_to_image)
colmap.process()
