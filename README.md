# ecCZ870 & ecFB
## Overview

ecCZ870 is an enzyme-constrained metabolic model of *Corynebacterium glutamicum* ATCC 14067. This repository provides notebooks and scripts for model construction, calibration, validation, and metabolic engineering target prediction using ecFB.

## Installation

### 1. Download the repository

Clone the repository and enter the project directory:

```bash
git clone https://github.com/zhangxinyuya/ecCZ870.git
cd ecCZ870
```

Alternatively, click **Code → Download ZIP** on the GitHub repository page, extract the archive, and open a terminal in the extracted project directory.

### 2. Create a Python environment

Using your installed Python, create a dedicated virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

The `requirements.txt` file is included in the repository. Install the listed packages by running:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run these commands from the project root directory, where `requirements.txt` is located.

### 4. Start JupyterLab

Install JupyterLab if it is not already available:

```bash
python -m pip install jupyterlab
```

Launch it from the project root directory:

```bash
python -m jupyterlab
```

Open the notebook you wish to run and select the Python environment in which the dependencies were installed.

## Usage

Keep the repository directory structure unchanged. The notebooks use relative paths to access model files, data, and local scripts.

Start JupyterLab from the project root directory so that paths such as `./script/`, `./ecscript/`, and `./model_Calibration/` resolve correctly.

Before running an analysis:

1. Check the input model and data paths in the notebook.
2. Set the target product, model parameters, and output paths as needed.
3. Run the notebook cells in order, as later cells may reuse functions or variables defined earlier.

The `script/` and `ecscript/` directories contain local project modules. They are included in this repository and do not need to be installed separately with pip.
## Acknowledgements

### ECMpy 2.0
The enzyme-constrained model construction workflow is based on ECMpy 2.0 originally developed by Professor Mao Zhitao and Professor Ma Hongwu's research group.
**Original repository:** https://github.com/tibbdc/ECMpy.git
**Reference:** Mao, Z., Niu, J., Zhao, J., Huang, Y., Wu, K., Yun, L., Guan, J., Yuan, Q., Liao, X., Wang, Z., Ma, H., 2024. ECMpy 2.0: A Python package for automated construction and analysis of enzyme-constrained models. Synthetic and Systems Biotechnology 9, 494–502. https://doi.org/10.1016/j.synbio.2024.04.005

We thank Dr. Zhitao Mao, Dr. Hongwu Ma, and their research group for developing and sharing the ECMpy 2.0 workflow.

### iBridge Algorithm
The iBridge component of the ecFB workflow is based on the iBridge 
algorithm originally developed by Professor Sang Yup Lee's research group.
**Original repository:** https://github.com/kaistsystemsbiology/MEResource.
**Reference:** Kim, W.J., Lee, Y., Kim, H.U., Ryu, J.Y., Yang, J.E., Lee, S.Y., 2023. Genome-wide identification of overexpression and downregulation gene targets based on the sum of covariances of the outgoing reaction fluxes. Cell Systems 14, 990-1001.e5. https://doi.org/10.1016/j.cels.2023.10.005.

We are grateful to Professor Sang Yup Lee and his research group for generously sharing the iBridge algorithm and granting permission to adapt their code for this project.

## Citation

If you use the code for ecCZ870 construction, please cite:

Mao, Z., Niu, J., Zhao, J., Huang, Y., Wu, K., Yun, L., Guan, J., Yuan, Q., Liao, X., Wang, Z., Ma, H., 2024. ECMpy 2.0: A Python package for automated construction and analysis of enzyme-constrained models. Synthetic and Systems Biotechnology 9, 494–502. https://doi.org/10.1016/j.synbio.2024.04.005

If you use the code of ecFB, please cite:

Kim, W.J., Lee, Y., Kim, H.U., Ryu, J.Y., Yang, J.E., Lee, S.Y., 2023. Genome-wide identification of overexpression and downregulation gene targets based on the sum of covariances of the outgoing reaction fluxes. Cell Systems 14, 990-1001.e5. https://doi.org/10.1016/j.cels.2023.10.005

