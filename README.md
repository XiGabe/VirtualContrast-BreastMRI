# VirtualContrast-BreastMRI

**Toward Contrast-Free Breast MRI: Improving Cancer Detection with Virtual Contrast Enhanced Imaging**

Hongxi Chen, Sunwoo Kwak, Sanyukta Adap, Katerina Dodelzon, Paula Feldman, Mert R. Sabuncu.

Virtual contrast-enhanced breast MRI generation, cancer detection using NC/CE/VCE images, and patient-level contrast-sparing simulation.

## Installation

Tested with Python 3.10 and PyTorch 2.7.1. Install PyTorch for your CUDA version, then:

```bash
git clone https://github.com/XiGabe/VirtualContrast-BreastMRI.git
cd VirtualContrast-BreastMRI
pip install -e '.[test]'
pytest -q
```

## Reproduce the reported results

Prediction files are included, so Table 1 and Figure 2 can be regenerated without downloading MRI data or running models:

```bash
bash scripts/reproduce.sh
```

Results are saved to `outputs/results/`.

| Input / classifier | AUROC | AUPRC |
|---|---|---|
| NC / NC | 0.865 ± 0.007 | 0.831 ± 0.020 |
| VCE / CE | 0.866 ± 0.007 | 0.841 ± 0.005 |
| VCE / VCE | 0.883 ± 0.010 | 0.854 ± 0.022 |
| CE / CE | 0.945 ± 0.003 | 0.927 ± 0.007 |
| NC + VCE / NC + CE | 0.901 ± 0.005 | 0.874 ± 0.005 |
| NC + VCE / NC + VCE | 0.896 ± 0.002 | 0.870 ± 0.004 |
| NC + CE / NC + CE | 0.943 ± 0.010 | 0.924 ± 0.012 |

Fusion averages classifier logits. Results are mean ± SD across three classifier repeats. Routing compares random assignment with NC-confidence-based assignment, keeping each patient's breasts on the same pathway.

## Data preparation

Obtain [DUKE](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/), [MAMAMIA](https://github.com/LidiaGarrucho/MAMAMIA), and [ODELIA](https://arxiv.org/abs/2506.00474) from their respective sources. The fixed cohort and patient-level splits are in `reproducibility/cohort.csv`; source identifiers are in `reproducibility/source_mapping.csv`.

| Split | Patients | Breasts |
|---|---:|---:|
| Training | 968 | 1,888 |
| Validation | 197 | 375 |
| Test | 245 | 473 |

Preprocessing expects paired, spatially aligned bilateral NIfTI volumes: fat-suppressed NC and the first acquired CE phase. Conversion from source DICOM/parquet files is not included. `source_mapping.csv` includes the original ODELIA patient identifiers and parquet shard names.

Arrange the converted files as:

```text
data/nifti/{DUKE,MAMAMIA,ODELIA}/{source_patient_id}/{source_patient_id}_0.nii.gz
data/nifti/{DUKE,MAMAMIA,ODELIA}/{source_patient_id}/{source_patient_id}_1.nii.gz
```

Generate the input index and preprocess:

```bash
python scripts/make_sources.py --root data/nifti --output data/sources.csv
python -m virtualcontrast.preprocess --sources data/sources.csv \
  --output data/cache --divider-model /path/to/BreastDividerModel
```

Alternatively, provide `sources.csv` directly with columns `patient,nc,ce,divider`; paths are relative to the CSV. Use patient IDs from `cohort.csv`.

[BreastDivider](https://huggingface.co/Bubenpo/BreastDivider) provides the anatomical masks. To generate them, install `nnunetv2==2.7.0` separately and obtain the model folder. Existing masks can be supplied through the `divider` column, allowing `--divider-model` to be omitted. The model checksum is in `reproducibility/preprocessing_reference.json`.

Paired volumes share NC-derived cropping and normalization, are resampled to 0.8×0.8×2 mm, and are padded to `[S,280,336]`. The cache contains `nc`, `ce`, and `breast` arrays with geometry metadata.

## Inference

The weight bundle contains one generator and NC/CE/VCE classifiers for seeds 2026, 2027 and 2028. Download and verify the versioned weights once the release assets are available:

```bash
python scripts/download_weights.py --tag v1.0.0
# Verify locally supplied weights:
python scripts/download_weights.py --verify-only

python -m virtualcontrast.inference generate --cache data/cache --split test
python -m virtualcontrast.inference predict --cache data/cache --split test \
  --output outputs/predictions
python -m virtualcontrast.evaluate --predictions outputs/predictions --output outputs/results
python -m virtualcontrast.routing --predictions outputs/predictions --output outputs/results
```

The generator predicts the enhancement residual and reconstructs VCE with 50%-overlapping sliding windows. Use `--batch-size` and `--chunk-size` to adjust GPU memory use. Weight checksums are listed in `reproducibility/weights.json`.

## Training

Settings are in `configs/default.json`. The generator uses a five-stage 3D U-Net, 64×256×256 patches, and 10 epochs of AdamW optimization with cosine learning-rate decay from 1e-4 to 1e-5. Classifiers use DINOv2 ViT-S/14 with a slice transformer, train for up to 60 epochs, and select the checkpoint with the highest validation AUROC.

Prepare a separate cache at `data/training_cache` using the preprocessing command above with `--output data/training_cache`, then train:

```bash
python -m virtualcontrast.train_generator --cache data/training_cache --output outputs/generator
mkdir -p outputs/trained_weights
cp outputs/generator/generator.pt outputs/trained_weights/generator.pt
python -m virtualcontrast.inference generate --cache data/training_cache --split all \
  --weights outputs/trained_weights

for seed in 2026 2027 2028; do
  for modality in nc ce vce; do
    python -m virtualcontrast.train_classifier --cache data/training_cache \
      --modality "$modality" --seed "$seed" \
      --output "outputs/${modality}_${seed}"
    cp "outputs/${modality}_${seed}/best.pt" "outputs/trained_weights/${modality}_${seed}.pt"
  done
done

python -m virtualcontrast.inference predict --cache data/training_cache --split test \
  --weights outputs/trained_weights --output outputs/trained_predictions
python -m virtualcontrast.evaluate --predictions outputs/trained_predictions --output outputs/trained_results
python -m virtualcontrast.routing --predictions outputs/trained_predictions --output outputs/trained_results
```

Add `--resume` to resume training from the last completed epoch. For offline classifier initialization, supply `--dinov2-weights /path/to/dinov2_vits14_pretrain.pth`.

## Image examples

Figure 1 uses the cases and slices in `reproducibility/figure1.json`. Supply an evaluation mask CSV with columns `sample_id,path`, pointing to the corresponding tumor-mask NIfTI files:

```bash
python scripts/plot_examples.py --cache data/cache --tumor-sources data/example_masks.csv
```

## License

Original code is MIT. Third-party components, model weights and datasets retain their respective terms; see `THIRD_PARTY_NOTICES.md`.
