# Multimodal AI Detection for Math-Heavy Scientific Text

This is the accompanying repository for the paper [Multimodal AI Detection for Math-Heavy Scientific Text](https://arxiv.org/abs/XXXX.XXXXX), which detects AI-generated mathematical writing from three representations of a passage: its LaTeX source (LaTeX detector), the extracted text of the compiled PDF (text-based detector) and a rendered image (SnapJudge).

## Links

- **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- **Models:** [Benson555/SnapJudge on HuggingFace](https://huggingface.co/Benson555/SnapJudge)
- **Dataset:** [Benson555/SnapJudge on HuggingFace](https://huggingface.co/datasets/Benson555/SnapJudge)
- **Demo:** [nlplab.cs.duke.edu/snapjudge](https://nlplab.cs.duke.edu/snapjudge)

## Setup

```bash
pip install -r requirements.txt
```

## Training

Configs for the three detectors are in `configs/`. All three fine-tune [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) with LoRA. The effective batch size is 16 for the LaTeX and text-based detectors and 64 for SnapJudge.

### LaTeX detector

```bash
torchrun --nproc_per_node 4 scripts/train.py --config configs/latex.yaml
```

### Text-based detector

```bash
torchrun --nproc_per_node 4 scripts/train.py --config configs/text.yaml
```

### SnapJudge

```bash
torchrun --nproc_per_node 4 scripts/train_image.py --config configs/image.yaml
```

## Inference

Run inference on any HuggingFace dataset (remote or local) or on a `.csv`, `.jsonl` or `.parquet` file. The scripts add a column `score`, the probability that the passage is AI-generated. For the LaTeX and text-based detectors it is the mean over 512-token windows with stride 256.

```bash
hf download Benson555/SnapJudge --local-dir SnapJudge

python scripts/inference.py \
  --config configs/latex.yaml \
  --checkpoint SnapJudge/latex \
  --dataset Benson555/SnapJudge \
  --split test \
  --output latex_predictions.jsonl

python scripts/inference.py \
  --config configs/text.yaml \
  --checkpoint SnapJudge/text \
  --dataset Benson555/SnapJudge \
  --split test \
  --output text_predictions.jsonl

python scripts/inference_image.py \
  --config configs/image.yaml \
  --checkpoint SnapJudge/image \
  --dataset Benson555/SnapJudge \
  --split test \
  --output image_predictions.jsonl
```

## Thresholds

A passage is classified as AI-generated if `score >= threshold`.

| Detector | FPR 0.1% | FPR 0.05% | FPR 0.01% | FPR 0.005% | FPR 0.001% |
|---|---|---|---|---|---|
| LaTeX | 0.9986114501953125 | 0.9994887113571167 | 0.9999279816031456 | 0.9999589702010154 | 0.9999960217535496 |
| Text-based | 0.9995121955871582 | 0.9997367536425591 | 0.9999424217402937 | 0.9999660922855138 | 0.9999893260324001 |
| SnapJudge | 0.9788680946826935 | 0.9905976241827011 | 0.9981755863428113 | 0.9991849524974823 | 0.9998685571908948 |

## Evaluation

For proofs and prose separately, the script reports the detection rate of every generator family and the Frontier and Open averages at the threshold of the chosen false-positive rate (`--detector` is `latex`, `text` or `image`).

```bash
python scripts/eval/evaluate.py --predictions latex_predictions.jsonl --detector latex --fpr 0.1%
```

## Citation

If you use the code, dataset, or models in this repository, please cite our paper:

```bibtex
@misc{huang2026multimodal,
      title={Multimodal AI Detection for Math-Heavy Scientific Text},
      author={Yixuan Huang and Danish Pruthi and Bhuwan Dhingra},
      year={2026},
      eprint={XXXX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/XXXX.XXXXX},
}
```

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
