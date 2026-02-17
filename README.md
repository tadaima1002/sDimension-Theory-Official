# s-Dimension Theory: A Unified Framework for Neural Network Internal Process Analysis

This repository contains a series of papers developing **s-dimension theory**—a novel framework for tracking and analyzing internal process integrity in neural networks.

## Research Program Overview

s-dimension theory introduces two fundamental metrics:
- **s-dimension**: Computational depth (abstraction level)
- **Structural Debt**: Accumulated dimensional inconsistency

Through systematic experiments, we demonstrate that diverse failure modes (hallucination, catastrophic forgetting) share a common mechanism: **information phase transitions** driven by Debt accumulation.

## Papers

### Paper 1: Phase Transitions in Neural Inference (Published)
**Status**: ✅ Zenodo DOI: [link]  
**Finding**: Critical point α≈0.3 where information processing transitions from ordered to disordered state.

📂 [Code & Results](./paper1-noise-phase-transition/)

---

### Paper 2: Catastrophic Forgetting as Structural Collapse (In Progress)
**Status**: 🚧 Draft complete, experiments ongoing  
**Finding**: Catastrophic forgetting manifests as +22% Debt increase, correlating with accuracy collapse (r=-0.87).

📂 [Code & Results](./paper2-catastrophic-forgetting/)

---

### Paper 3: Unified Theory (Planned)
**Status**: 📅 Experiments starting Q1 2026  
**Goal**: Demonstrate that hallucination and catastrophic forgetting are manifestations of the same phase transition mechanism.

📂 [Code & Results](./paper3-unified-theory/)

---

## Quick Start
```bash
# Install dependencies
pip install torch torchvision numpy matplotlib scipy

# Run Paper 1 experiments
cd paper1-noise-phase-transition/code
python phase1_continuous_experiment.py

# Run Paper 2 experiments
cd ../../paper2-catastrophic-forgetting/code
python catastrophic_forgetting_experiment.py
```

## Theoretical Foundation

For mathematical proofs and theoretical background:
- 📄 [Multivalued Function Isomorphism](./theory/multivalued_isomorphism_proof.md)
- 📄 [s-Dimension Formal Definition](./theory/sdimension_definition.md)

## Citation

If you use this work, please cite:
```bibtex
@software{Imamura2026_sDimTheory,
  author = {Imamura},
  title = {s-Dimension Theory: A Unified Framework for Neural Network Internal Process Analysis},
  year = {2026},
  url = {https://github.com/yourusername/sdimension-theory-official}
}
```

## License

Apache 2.0

## Contact

**Imamura** (Independent Researcher) 
GitHub ID:tadaima1002 
Contact: (Available via GirHub)