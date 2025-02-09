#!/bin/bash

# Function to download and unzip files
download_and_unzip() {
  local url=$1
  local dest_dir=$2
  local filename
  filename=$(basename "$url")
  
  echo "Downloading $filename into $dest_dir..."
  wget -P "$dest_dir" "$url"
  
  echo "Unzipping $filename in $dest_dir..."
  unzip -o -d "$dest_dir" "$dest_dir/$filename"
}

# Function to simply download a file (non-zip)
download_file() {
  local url=$1
  local dest_dir=$2
  echo "Downloading $(basename "$url") into $dest_dir..."
  wget -P "$dest_dir" "$url"
}

echo "Downloading datasets for secondary structure prediction tasks..."
download_and_unzip "https://zenodo.org/records/14795554/files/data_ss.zip" "downstream_ss"

echo "Downloading datasets and head weights for protein-RNA interaction task..."
download_and_unzip "https://zenodo.org/records/14795554/files/data_rbp.zip" "downstream_rbp"
download_and_unzip "https://zenodo.org/records/14795554/files/out_rbp.zip" "downstream_rbp/exp/prismnet_Hela_eval"

echo "Downloading dataset and head weights for mean ribosome loading task..."
download_and_unzip "https://zenodo.org/records/14840194/files/data_mrl.zip" "downstream_mrl"
download_file "https://zenodo.org/records/14795554/files/mrlHead.ckpt" "weights"

echo "All downloads are complete."
