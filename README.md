# Quality Analysis for Machine Learning models

The three quality pillars are:

- **Functionality:** groups analyses that evaluate how ”well“ an AI module performs a
given task (i.e. assessing the suitability of an AI module for an application domain).


- **Comprehensibility:** groups analyses that try to open the blackbox and enable
stakeholders (model producers, users, or regulators) to interpret decisions and the
decision-making process. 


- **Robustness:** groups analyses that assess how the ML component responds to small
changes in the input. 


The latest release performs functionality and robustness analyses for image classification 
and regression models. The next versions will include comprehensibility analysis and accept text data.

# Installation

Using the PyPi package:
```
pip install ml-model-quality-analysis
```

# Notable Dependencies

TensorFlow 2.3 is required:
```
pip install tensorflow==2.3
```

# Getting Started

For examples on performing quality analysis for ML models, see the [Quality Report Notebook.](https://github.com/mariagrandury/ml-model-quality-analysis/blob/main/quality_report.ipynb)
