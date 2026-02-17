
# sDimension Theory: Information Phase Transition in Neural Networks
[![DOI](https://zenodo.org/badge/1149524619.svg)](https://doi.org/10.5281/zenodo.18646643)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)


> **Discovery of Critical Threshold α ≈ 0.3 with p < 4.72×10⁻¹⁹⁸ Statistical Significance**

---

## 🔬 Abstract

We report the **first empirical discovery** of an information phase transition in neural network inference. Using a novel "Dimensional Debt" metric from sDimension theory, we demonstrate that neural networks undergo catastrophic structural collapse when input noise exceeds a critical threshold **α ≈ 0.3**.

**Key Findings**:
- 📊 **Statistical significance**: p < 4.72×10⁻¹⁹⁸ (astronomically significant)
- 📈 **Effect size**: Cohen's d = 8.02 (exceptionally large, ~6.7× standard threshold)
- 🔗 **Correlation**: r = 0.966 (near-perfect linearity)
- 🎯 **Classification**: AUC = 1.000 (perfect separation)

This phase transition is analogous to physical phase transitions (ice→water→vapor) and has immediate implications for:
- 🚨 Hallucination detection in AI systems
- ✅ Model reliability assessment
- 🔄 Continual learning

---

## 📄 Research Papers

### English Version
- **Paper**: [sdimension_preprint_v1.md](sdimension_preprint_v1.md)
- **PDF**: [sdimension_preprint_v1.pdf](sdimension_preprint_v1.pdf)

### 日本語版 (Japanese Version)
- **論文**: [sdimension_preprint_v1.0_jp.md](sdimension_preprint_v1.0_jp.md)
- **PDF**: [sdimension_preprint_v1.0_jp.pdf](sdimension_preprint_v1.0_jp.pdf)

---

## 🧪 Experimental Code

### Phase 1: Noise Mixing Continuous Experiment
- **Code**: [phase1_continuous_experiment.py](phase1_continuous_experiment.py)
- **Results**: [phase1_experiment.png](phase1_experiment.png)

### Quick Start

```bash
# Clone repository
git clone https://github.com/tadaima1002/sDimension-Theory-Official.git
cd sDimension-Theory-Official

# Install dependencies
pip install torch torchvision numpy matplotlib scipy scikit-learn --break-system-packages

# Run experiment (~30 minutes on GTX 1050 Ti)
python phase1_continuous_experiment.py
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 4GB+ VRAM (tested on GTX 1050 Ti)

---

## 📊 Main Results

![Phase Transition Discovery](phase1_experiment.png)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **p-value** | 4.72×10⁻¹⁹⁸ | Beyond astronomical significance |
| **Cohen's d** | 8.02 | Exceptionally large effect size |
| **Pearson r** | 0.966 | Near-perfect linear correlation |
| **AUC** | 1.000 | Perfect classification accuracy |
| **Critical threshold** | α ≈ 0.3 | Information phase transition point |

---

## 🎯 Key Insights

### 1. What is sDimension Theory?

sDimension theory tracks **computational history** and **structural integrity** of neural network values:

```
Every value: Ψ = (v, s, d)
  v: numerical value
  s: sDimension (computational depth)
  d: Dimensional Debt (structural mismatch)
```

### 2. What is Dimensional Debt?

When different computational paths merge (e.g., residual connections), structural "debt" accumulates:

```python
# At residual merge:
gap = |s_main - s_shortcut|  # depth mismatch
d_new = d_main + d_shortcut + gap  # debt accumulates
```

**High debt = unreliable computation**, even if output looks confident.

### 3. Phase Transition at α ≈ 0.3

| Phase | Noise ratio α | Debt | State |
|-------|---------------|------|-------|
| **Ordered** | 0.0 - 0.3 | Low (d<5) | Information preserved |
| **Critical** | ~0.3 | Rapid increase | Phase transition |
| **Disordered** | 0.3 - 1.0 | High (d>15) | Structural collapse |

---

## 🚀 Applications

### Hallucination Detection
```python
if debt > threshold:
    warning("Output is structurally unreliable")
    # Don't trust high confidence scores
```

### Model Monitoring
- Track structural integrity during deployment
- Detect adversarial attacks
- Monitor out-of-distribution inputs

### Architecture Design
- Understand why skip connections work
- Design robust deep architectures
- Prevent catastrophic forgetting

---

## 📖 Citation

```bibtex
@article{imamura2026sdimension,
  title={Information Phase Transition in Neural Networks: Discovery of Critical Threshold via sDimension Theory},
  author={Imamura},
  year={2026},
  month={February},
  publisher={GitHub},
  url={https://github.com/tadaima1002/sDimension-Theory-Official},
  note={Preprint. DOI pending via Zenodo integration}
}
```

---

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This research was conducted independently without institutional affiliation. Special thanks to the Claude AI system for technical discussions during theory development.

---

## 📧 Contact

- **GitHub**: [@tadaima1002](https://github.com/tadaima1002)
- **Issues**: [Open an issue](https://github.com/tadaima1002/sDimension-Theory-Official/issues)
- **Discussions**: [Start a discussion](https://github.com/tadaima1002/sDimension-Theory-Official/discussions)

---

## 🔗 Links

- **Zenodo Archive**: Coming soon (DOI will be assigned upon Zenodo integration)
- **arXiv**: Planned submission after peer feedback

---

**Last Updated**: February 15, 2026  
**Version**: 1.0
