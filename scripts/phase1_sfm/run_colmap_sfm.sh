#!/bin/bash
# Phase 1 Step 1: COLMAP SfM Pipeline (COLMAP 4.x compatible)
# Runs COLMAP feature extraction, matching, and sparse reconstruction
# on the Collabolab images using the COLMAP Docker container with GPU support.

set -e

WORKSPACE="/workspace"
IMAGE_DIR="${WORKSPACE}/images"
DATABASE="${WORKSPACE}/sfm_output/database.db"
OUTPUT_DIR="${WORKSPACE}/sfm_output/sparse"

echo "============================================"
echo "Phase 1: COLMAP SfM Sparse Reconstruction"
echo "  COLMAP 4.x / RTX 4090 / 92 images"
echo "============================================"
echo ""

# Create output directories
mkdir -p "${WORKSPACE}/sfm_output/sparse"

# Count images
NUM_IMAGES=$(ls ${IMAGE_DIR}/*.jpg 2>/dev/null | wc -l)
echo "Found ${NUM_IMAGES} images"
echo ""

# ============================================
# Step 1: Feature Extraction (GPU SIFT)
# ============================================
echo "[1/4] Feature Extraction (GPU SIFT)..."
colmap feature_extractor \
    --database_path "${DATABASE}" \
    --image_path "${IMAGE_DIR}" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 1 \
    --FeatureExtraction.max_image_size 3024 \
    --SiftExtraction.max_num_features 8192 \
    --SiftExtraction.first_octave -1

echo "[1/4] Feature Extraction DONE"
echo ""

# ============================================
# Step 2: Feature Matching (Exhaustive with GPU)
# ============================================
echo "[2/4] Exhaustive Feature Matching..."
colmap exhaustive_matcher \
    --database_path "${DATABASE}" \
    --FeatureMatching.use_gpu 1 \
    --FeatureMatching.max_num_matches 32768 \
    --SiftMatching.max_ratio 0.8 \
    --SiftMatching.max_distance 0.7

echo "[2/4] Feature Matching DONE"
echo ""

# ============================================
# Step 3: Incremental Mapper (Sparse Reconstruction)
# ============================================
echo "[3/4] Incremental Mapping (Sparse Reconstruction)..."
colmap mapper \
    --database_path "${DATABASE}" \
    --image_path "${IMAGE_DIR}" \
    --output_path "${OUTPUT_DIR}" \
    --Mapper.ba_global_max_num_iterations 50 \
    --Mapper.ba_global_max_refinements 5 \
    --Mapper.min_num_matches 15 \
    --Mapper.init_min_num_inliers 100 \
    --Mapper.abs_pose_min_num_inliers 30 \
    --Mapper.abs_pose_min_inlier_ratio 0.25 \
    --Mapper.max_reg_trials 3

echo "[3/4] Incremental Mapping DONE"
echo ""

# ============================================
# Step 4: Export as TXT for easier parsing
# ============================================
echo "[4/4] Exporting model as TXT..."
RECON_DIR=$(ls -d ${OUTPUT_DIR}/*/ 2>/dev/null | head -1)
if [ -z "${RECON_DIR}" ]; then
    echo "ERROR: No reconstruction found!"
    exit 1
fi

TXT_OUTPUT="${WORKSPACE}/sfm_output/sparse_txt"
mkdir -p "${TXT_OUTPUT}"

colmap model_converter \
    --input_path "${RECON_DIR}" \
    --output_path "${TXT_OUTPUT}" \
    --output_type TXT

echo "[4/4] Export DONE"
echo ""

# ============================================
# Print summary
# ============================================
echo "============================================"
echo "SfM Reconstruction Summary"
echo "============================================"
colmap model_analyzer \
    --path "${RECON_DIR}" 2>&1 || true

echo ""
echo "Output files:"
ls -la "${TXT_OUTPUT}/"
echo ""
echo "============================================"
echo "SfM pipeline completed successfully!"
