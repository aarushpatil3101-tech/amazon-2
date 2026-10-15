# Amazon Inventory Reconciliation Using AI

UE24CS352A Machine Learning mini-project - **Aarush Patil (PES1UG24AM005)**, PES University.

Amazon fulfilment-centre bins sometimes hold a different number of items than the inventory system
says. This project **counts the items in a bin from one photo** (0-5 items, Amazon Bin Image Dataset)
and uses the predicted count to **flag bins whose recorded inventory disagrees with the camera**.

## Deliverables in this repo

| What | Where |
|---|---|
| Source code | `*.py`, `abid/`, `notebooks/colab_run.ipynb` |
| 2-page write-up (PDF) | `report/report.pdf` |
| Review slides | `slides/slides.pptx` |
| Raw results and figures | `results/` (`results.json`, `baselines.json`, `fig_*.png`) |
| Per-run metrics, histories, test predictions | `runs/r34_cls/`, `runs/r34_ord/` |

## Method

| Stage | What we do |
|---|---|
| Data | Random 30,000-image subset of ABID (bins with 0-5 items), resized to 224x224, stratified 70/20/10 split (21,000 / 6,000 / 3,000) |
| Baselines | Random, majority class, Logistic Regression / Linear SVM / Decision Tree / RBF-SVM on PCA(256) of 64x64 raw pixels, RBF-SVM on HOG |
| CNN | ImageNet-pretrained **ResNet34**: head-only stage, then full fine-tuning with discriminative LRs (lr/9, lr/3, lr), AdamW, **SGDR** cosine warm restarts, LR-range test, h-flip augmentation, h-flip test-time augmentation |
| Ablation | **Ordinal head** (5 cumulative binary outputs, BCE) vs 6-way softmax |
| Reconciliation | Simulated recorded-inventory errors (15% of bins off by +-1/2) -> precision / recall / F1 / AUROC of mismatch detection |
| Demo | Gradio app + CLI: upload a bin photo, enter the recorded count, get MATCH / VARIANCE |

## Results (test split, 3,000 images, one run each)

| Model | Test acc. | RMSE | Within 1 item |
|---|---|---|---|
| Random guess | 16.7% | - | - |
| Majority class | 25.3% | 1.35 | 68.8% |
| Logistic regression (PCA) | 28.5% | 1.37 | 73.1% |
| Linear SVM (PCA) | 27.7% | 1.37 | 73.0% |
| Decision tree (PCA) | 25.5% | 1.57 | 65.7% |
| RBF-SVM (PCA) | 29.8% | 1.32 | 74.9% |
| RBF-SVM (HOG) | 31.9% | 1.32 | 75.3% |
| **ResNet34, softmax head + TTA** | **41.2%** | 1.11 | 83.4% |
| ResNet34, ordinal head + TTA | 39.4% | **1.04** | 86.6% |

* The softmax ResNet34 is +29% (relative) over the best classical model and 2.5x random guessing.
* The ordinal head lowers RMSE (1.04 vs 1.11) but loses 1.8 points of accuracy.
* Variance detection (15% simulated recording errors): precision 19.4%, recall 81.4%, F1 0.31, AUROC 0.73.
* Trained on a Colab T4 in about 15 minutes per model; 6.2 ms per image at batch size 1.
* Stanford's published ResNet34 (56.2% acc, RMSE 0.90) used a larger, different subset and is higher than ours.

Full numbers: `results/results.json`. Figures: `results/fig_*.png`.

## Repository layout

```
download_data.py     download a random ABID subset over HTTPS (no AWS account), resumable
build_splits.py      stratified 70/20/10 split -> data/splits.csv
baselines.py         classical baselines (PCA / HOG + LogReg / SVM / tree)
train_cnn.py         ResNet transfer learning (cls or ordinal head, LR finder, SGDR, TTA)
make_results.py      aggregate runs -> results/results.json + figures
demo.py              single image / batch CSV / Gradio web demo
abid/                shared code (metrics, features, LR finder, variance simulation)
notebooks/           colab_run.ipynb - the whole pipeline on a free T4
tests/               make_synthetic.py - tiny synthetic dataset for smoke tests
report/  slides/     final write-up and slide deck
results/  runs/      outputs of the experiments reported above
```

## Reproduce (Google Colab, free T4)

1. Open `notebooks/colab_run.ipynb` in Colab, set Runtime -> T4 GPU, upload this repo as a zip when asked.
2. Run the cells top to bottom (about 1-1.5 h).

Or manually:

```bash
pip install -r requirements.txt
python download_data.py --n 30000 --out data
python build_splits.py --data data
python baselines.py --data data --out results/baselines.json
python train_cnn.py --data data --arch resnet34 --head cls     --name r34_cls --lr-find
python train_cnn.py --data data --arch resnet34 --head ordinal --name r34_ord --lr-find
python make_results.py --data data
```

## Demo

Model weights (`runs/<name>/best.pt`, about 83 MB) are not stored in this repo; re-run the training
step (or copy the file from your Colab/Drive) first.

```bash
python demo.py --run runs/r34_cls --gradio                                  # web UI
python demo.py --run runs/r34_cls --image some_bin.jpg --recorded 3          # CLI
python demo.py --run runs/r34_cls --csv recorded.csv --images data/images    # csv: id,recorded_qty
```

## Data

Amazon Bin Image Dataset: <https://registry.opendata.aws/amazon-bin-imagery/> (bucket `aft-vbi-pds`;
`bin-images/<id>.jpg`, `metadata/<id>.json`, label = `EXPECTED_QUANTITY`).

## Limitations

* Many bins are occluded (tape, bags); some cannot be counted even by a human.
* The LR finder picked a very low head LR for both CNN runs (2.9e-5 softmax, 1e-5 ordinal; backbone
  gets 1/3 and 1/9 of it), so the networks were probably under-trained. A fixed LR near 1e-3 without
  `--lr-find` is the first thing to try to close the gap to the published 56%.
* One random split and one seed per model; baselines use 64x64 pixels and an 8K-image SVM subsample.
* Recorded-inventory errors are simulated; the public data only has true counts.
