# Paper 2: Visualizing Catastrophic Forgetting Mechanism

## Overview

This paper demonstrates that catastrophic forgetting is not merely "weight overwriting" but a measurable **structural collapse** quantified by Debt accumulation.

## Key Results

- Debt increases +22% during forgetting
- Strong negative correlation (r=-0.87) with retained accuracy
- First internal diagnostic for catastrophic forgetting

## Reproduce Results
```bash
python code/catastrophic_forgetting_experiment.py
```

## Files

- `paper/paper2_draft.md`: Full paper draft
- `code/catastrophic_forgetting_experiment.py`: Main experiment
- `results/`: Experimental data

## Status

🚧 Draft complete, awaiting additional experiments

## Related

- See [Paper 1](../paper1-noise-phase-transition/) for phase transition background
- Theory: [s-Dimension Definition](../theory/sdimension_definition.md)
```

---