# DrumlinDistributionFiltering
Python analysis of velocity and thickness distributions associated with matched drumlin observations.

# DrumlinDistributionFiltering

This repository contains the Python scripts and input data used to analyse
ice velocity and ice thickness distributions associated with matched
observations in the accompanying manuscript - Physical conditions for drumlin formation from matching drumlin occurrence with ice sheet model simulations. Veness et al., 2026/7.

## Contents

- `velocity_analysis.py`: velocity histogram filtering, distribution fitting,
  statistical comparisons, and cumulative probability plots.
- `thickness_analysis.py`: thickness histogram filtering, distribution fitting,
  statistical comparisons, and cumulative probability plots.
- `data/`: input CSV files used by the analysis.
- `output/`: output figures created when the scripts are run.

## Input data

The scripts expect the following files:

```text
data/VelocitySample.csv
data/VelocityBaseline.csv
data/ThicknessSample.csv
data/ThicknessBaseline.csv

## Citation

If you use this code, please cite the associated manuscript and the archived software release.

Full citation and DOI information will be added following publication.

## Licence

This code is made available under the MIT License. 
