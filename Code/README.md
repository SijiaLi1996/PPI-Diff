```markdown
# PPI-Diff: Resolution-Aware Fusion of Aligned PDB Fragments for PPI Structure and Sequence Co-Design

This repository contains the official implementation of **PPI-Diff**, a novel conditional angle diffusion model for the co-design of protein-protein interaction (PPI) structures and sequences. 

##  Overview
PPI-Diff utilizes a resolution-aware feature fusion mechanism, incorporating ESM-2 language model embeddings to generate highly accurate *de novo* peptide binders. This repository provides both the high-throughput benchmark generation scripts and an easy-to-deploy graphical Web Server interface.

##  Data and Model Weights Availability
Due to file size limits on GitHub, the pre-trained model weights and large benchmark datasets are hosted on Google Drive. 

 **[Download Weights and Datasets Here (Google Drive)](https://drive.google.com/drive/folders/1tRKfKX6zVU4HKoqzxIVgmOAaw3cmZMVZ?usp=drive_link)**

**Preparation:**
After downloading the files from the Google Drive link above, please organize them as follows:
1. Place the model weights file (`checkpoint_epoch_500.pth`) inside the `Web/PPI_Diff/` directory.
2. Extract the dataset files (`protein_features_by_ppi200.zip` and `ppi_pdb_by_uniprot.zip`) to your local directory for benchmark testing.

##  Environment Setup
To run the code, please install the following dependencies. We recommend using an isolated Conda environment.

```bash
# Core dependencies
pip install torch pandas numpy tqdm flask werkzeug

# Install ESM for language model embeddings
pip install fair-esm

```

##  Deploying the Web Server Locally

For security and stability, we provide the complete Web Server package for local deployment. You can easily run the PPI-Diff graphical interface on your own machine.

1. Navigate to the `Web` directory:
```bash
cd Web

```


2. Ensure the model weights (`checkpoint_epoch_500.pth`) are correctly placed in `Web/PPI_Diff/`.
3. Start the Flask server:
```bash
python app.py

```


4. Open your web browser and navigate to `http://127.0.0.1:5000`. You can now upload a `.npz` target feature file to instantly generate the binder's 3D structure (`.pdb`) and sequence (`.fasta`) through the graphical interface.

##  Running Batch Benchmark Generation

For high-throughput benchmarking on large datasets, use the provided generation script. Make sure your datasets downloaded from Google Drive are unzipped.

```bash
python generate_benchmark.py \
  --inputs_file path/to/human_protein_interactions_verified_200.tsv \
  --features_dir path/to/protein_features_by_ppi200 \
  --pdb_dir path/to/ppi_pdb_by_uniprot \
  --checkpoint path/to/checkpoint_epoch_500.pth \
  --output_dir ./benchmark_results

```

##  Citation

If you find this code or our data useful in your research, please consider citing our work:

```bibtex
@article{PPI_Diff_2026,
  title={Resolution-Aware Fusion of Aligned PDB Fragments for PPI Structure and Sequence Co-Design},
  author={Li, Sijia and Dong, Benzhi and Xu, Dali and Wang, Guohua},
  journal={Biomolecules},
  year={2026}
}

```