[<img src="LLM" align="right" width="25%" padding-right="350">]()

# `EARTHQUAKE_PREDICTOR.GIT`

#### <code>❯ REPLACE-ME</code>

<p align="left">
	<img src="https://img.shields.io/github/license/lhhhappy/Earthquake_predictor.git?style=flat-square&logo=opensourceinitiative&logoColor=white&color=00a1ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/lhhhappy/Earthquake_predictor.git?style=flat-square&logo=git&logoColor=white&color=00a1ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/lhhhappy/Earthquake_predictor.git?style=flat-square&color=00a1ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/lhhhappy/Earthquake_predictor.git?style=flat-square&color=00a1ff" alt="repo-language-count">
</p>
<p align="left">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=flat-square&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=flat-square&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/JSON-000000.svg?style=flat-square&logo=JSON&logoColor=white" alt="JSON">
</p>

<br>

##### 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📂 Repository Structure](#-repository-structure)
- [🧩 Modules](#-modules)
- [🚀 Getting Started](#-getting-started)
    - [🔖 Prerequisites](#-prerequisites)
    - [📦 Installation](#-installation)
    - [🤖 Usage](#-usage)
    - [🧪 Tests](#-tests)
- [📌 Project Roadmap](#-project-roadmap)
- [🤝 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

<code>❯ REPLACE-ME</code>

---

## 👾 Features

<code>❯ REPLACE-ME</code>

---

## 📂 Repository Structure

```sh
└── Earthquake_predictor.git/
    ├── README.md
    ├── data_preprocess_pipeline
    │   ├── __pycache__
    │   │   └── utils.cpython-310.pyc
    │   ├── pipeline.py
    │   └── utils.py
    ├── dataset
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   └── dataset_utils.cpython-310.pyc
    │   ├── dataset_test.ipynb
    │   └── dataset_utils.py
    ├── experiment
    │   ├── data_preprocess_ipynb
    │   │   ├── data_preposs_earthquake.ipynb
    │   │   ├── data_preposs_gnss.ipynb
    │   │   ├── dataset.ipynb
    │   │   ├── earthquake_usgs.ipynb
    │   │   └── plot_GNSS.ipynb
    │   └── data_preprocess_py
    │       ├── down_earthquake_data.py
    │       ├── download_GNSS_data.py
    │       ├── get_aij.py
    │       ├── log_energy.py
    │       └── usgs_data_area_save.py
    ├── loss
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   ├── loss.cpython-310.pyc
    │   │   └── loss_utils.cpython-310.pyc
    │   ├── loss_test.ipynb
    │   └── loss_utils.py
    ├── model
    │   ├── ES_net.py
    │   ├── Earthquake_net.ipynb
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── ES_net.cpython-310.pyc
    │       └── __init__.cpython-310.pyc
    ├── model_params.json
    ├── station_dict_all.pkl
    ├── task
    │   └── train_es_net.sh
    └── train.py
```

---

## 🧩 Modules

<details closed><summary>.</summary>

| File | Summary |
| --- | --- |
| [model_params.json](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/model_params.json) | <code>❯ REPLACE-ME</code> |
| [train.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/train.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>dataset</summary>

| File | Summary |
| --- | --- |
| [dataset_test.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/dataset/dataset_test.ipynb) | <code>❯ REPLACE-ME</code> |
| [dataset_utils.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/dataset/dataset_utils.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>experiment.data_preprocess_py</summary>

| File | Summary |
| --- | --- |
| [usgs_data_area_save.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_py/usgs_data_area_save.py) | <code>❯ REPLACE-ME</code> |
| [down_earthquake_data.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_py/down_earthquake_data.py) | <code>❯ REPLACE-ME</code> |
| [log_energy.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_py/log_energy.py) | <code>❯ REPLACE-ME</code> |
| [get_aij.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_py/get_aij.py) | <code>❯ REPLACE-ME</code> |
| [download_GNSS_data.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_py/download_GNSS_data.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>experiment.data_preprocess_ipynb</summary>

| File | Summary |
| --- | --- |
| [earthquake_usgs.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_ipynb/earthquake_usgs.ipynb) | <code>❯ REPLACE-ME</code> |
| [data_preposs_gnss.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_ipynb/data_preposs_gnss.ipynb) | <code>❯ REPLACE-ME</code> |
| [data_preposs_earthquake.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_ipynb/data_preposs_earthquake.ipynb) | <code>❯ REPLACE-ME</code> |
| [plot_GNSS.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_ipynb/plot_GNSS.ipynb) | <code>❯ REPLACE-ME</code> |
| [dataset.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/experiment/data_preprocess_ipynb/dataset.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>model</summary>

| File | Summary |
| --- | --- |
| [Earthquake_net.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/model/Earthquake_net.ipynb) | <code>❯ REPLACE-ME</code> |
| [ES_net.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/model/ES_net.py) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>task</summary>

| File | Summary |
| --- | --- |
| [train_es_net.sh](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/task/train_es_net.sh) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>loss</summary>

| File | Summary |
| --- | --- |
| [loss_utils.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/loss/loss_utils.py) | <code>❯ REPLACE-ME</code> |
| [loss_test.ipynb](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/loss/loss_test.ipynb) | <code>❯ REPLACE-ME</code> |

</details>

<details closed><summary>data_preprocess_pipeline</summary>

| File | Summary |
| --- | --- |
| [utils.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/data_preprocess_pipeline/utils.py) | <code>❯ REPLACE-ME</code> |
| [pipeline.py](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/data_preprocess_pipeline/pipeline.py) | <code>❯ REPLACE-ME</code> |

</details>

---

## 🚀 Getting Started

### 🔖 Prerequisites

**Python**: `version x.y.z`

### 📦 Installation

Build the project from source:

1. Clone the Earthquake_predictor.git repository:
```sh
❯ git clone https://github.com/lhhhappy/Earthquake_predictor.git
```

2. Navigate to the project directory:
```sh
❯ cd Earthquake_predictor.git
```

3. Install the required dependencies:
```sh
❯ pip install -r requirements.txt
```

### 🤖 Usage

To run the project, execute the following command:

```sh
❯ python main.py
```

### 🧪 Tests

Execute the test suite using the following command:

```sh
❯ pytest
```

---

## 📌 Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## 🤝 Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/lhhhappy/Earthquake_predictor.git/issues)**: Submit bugs found or log feature requests for the `Earthquake_predictor.git` project.
- **[Submit Pull Requests](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/lhhhappy/Earthquake_predictor.git/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/lhhhappy/Earthquake_predictor.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/lhhhappy/Earthquake_predictor.git/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=lhhhappy/Earthquake_predictor.git">
   </a>
</p>
</details>

---

## 🎗 License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## 🙌 Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
