# Fixing GAN Mode Collapse with Spectral Normalization for E-commerce Product Image Generation

**CAC Assignment: Generative Model Architectures** · Shreya · Master of Data Science, CHRIST (Deemed to be University), Bengaluru

The assignment: study four generative architectures (Transformer, GAN, VAE, Diffusion), pick one disadvantage, fix it, prove the fix with numbers, and use the fixed model in a real-world app.

| Phase | What I did | Where |
|---|---|---|
| 1. Comparative study | Working, advantages and disadvantages of all four architectures, compared on training stability, output quality, generation speed and latent-space interpretability. Disadvantages taken from each original paper where the authors stated them. | [`report/CAC_Report.pdf`](report/CAC_Report.pdf), Section 2 |
| 2. Bottleneck selection | **GAN → mode collapse**, the "Helvetica scenario" named in Goodfellow et al. (2014). Fix: **spectral normalization** on every discriminator layer + **hinge loss**. | Report, Section 3 |
| 3. Implementation & benchmarking | Baseline DCGAN vs improved GAN, same generator, data and settings. Metrics: FID, precision/recall, categories covered, KL divergence, inference latency. | [`src/`](src), [`results/`](results), Report Section 4 |
| 4. Deployment | **Fashion Design Idea Generator** in Gradio (improved model only): generate by category, "more like this", mix two designs, download a mood board. | [`app.py`](app.py), Report Section 5 |

## Results

> Fill this table from `results/metrics.csv` after running the notebook.

| Metric (10,000 samples) | Baseline DCGAN | Improved (SN + hinge) |
|---|---|---|
| FID ↓ | – | – |
| Precision ↑ | – | – |
| Recall ↑ | – | – |
| Categories covered (of 20) ↑ | – | – |
| KL(real ‖ generated categories) ↓ | – | – |
| Latency, GPU batch 1 (ms/img) | – | – |

<!-- After training, uncomment:
![Baseline (left) vs improved (right), same noise](results/side_by_side.png)
![Category distribution](results/category_distribution.png)
![Training curves](results/training_curves.png)
-->

## The idea in one paragraph

A GAN's generator only learns through the discriminator's gradients. If the discriminator becomes too sharp, those gradients vanish everywhere except near the current fakes, and the generator piles onto whatever currently fools it. That's mode collapse. Spectral normalization divides each discriminator weight matrix by its largest singular value, so the discriminator is 1-Lipschitz and its score changes smoothly across the whole image space. The generator keeps getting useful gradients towards parts of the data it isn't covering yet. The hinge loss stops the discriminator over-pushing easy samples and doesn't saturate. **Only the discriminator changes, so the generator used in the app is exactly as fast as before.**

## Dataset

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small): 44,072 Myntra catalogue photos, loaded from the Hugging Face mirror `ashraq/fashion-product-images-small`. I keep the 20 most common article types (Tshirts, Shirts, Kurtas, Watches, Casual Shoes, …), pad each image to a white square and resize to 64×64. The 20 categories are the "modes" used to measure collapse: a ResNet-18 classifier labels every generated image, and I count which categories each GAN actually produces.

## Run it (Google Colab, free T4)

1. Open [`notebooks/CAC_GAN_Colab.ipynb`](notebooks/CAC_GAN_Colab.ipynb) in Colab and set **Runtime → T4 GPU**.
2. Put this repo's URL in the setup cell and run all cells. Data, checkpoints and logs are saved to Google Drive; if Colab disconnects, re-run and training resumes.
3. The last cell launches the demo with a public link.

Or from a terminal with a GPU:

```bash
pip install -r requirements.txt
python -m src.data --out data/fashion64.npz --top_k 20
python -m src.train --model baseline --epochs 40
python -m src.train --model improved --epochs 40
python -m src.classifier
python -m src.evaluate
cp runs/improved/G_final.pt checkpoints/G_improved.pt
python app.py            # add --share for a public link
```

To rebuild the report with the latest results: `cd report && npm install docx && node build_report.js`.

## Repository structure

```
├── app.py                      Gradio demo (improved model only)
├── src/
│   ├── data.py                 download + preprocess (64×64, top-20 categories)
│   ├── models.py               shared DCGAN generator; D with BatchNorm or spectral norm
│   ├── losses.py               BCE (baseline) and hinge (improved)
│   ├── train.py                training with checkpoints, resume, per-epoch logs
│   ├── classifier.py           ResNet-18 category classifier (metric + app filter)
│   └── evaluate.py             FID, precision/recall, coverage, latency, figures
├── notebooks/CAC_GAN_Colab.ipynb
├── results/                    metrics.json/csv, figures, training logs
├── checkpoints/                G_improved.pt, G_baseline.pt, classifier.pt
└── report/                     CAC_Report.docx / .pdf + build_report.js
```

## Limitations

64×64 images are thumbnails, good for ideas on colour, pattern and silhouette but not for product photos. Each model was trained with one seed because of Colab time limits. The category filter in the app depends on the classifier.

## Key references

- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014
- Radford et al., *Unsupervised Representation Learning with Deep Convolutional GANs*, ICLR 2016
- Miyato et al., *Spectral Normalization for Generative Adversarial Networks*, ICLR 2018
- Lim & Ye, *Geometric GAN*, arXiv 2017
- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule…* (FID), NeurIPS 2017
- Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models*, NeurIPS 2019

Full reference list in the report.
