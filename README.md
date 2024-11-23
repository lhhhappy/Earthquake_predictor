</p>
<p align="center"><h1 align="center">EARTHQUAKE_PREDICTOR.GIT</h1></p>
<p align="center">
	<em><code>❯ REPLACE-ME</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/lhhhappy/Earthquake_predictor.git?style=default&logo=opensourceinitiative&logoColor=white&color=c59adc" alt="license">
	<img src="https://img.shields.io/github/last-commit/lhhhappy/Earthquake_predictor.git?style=default&logo=git&logoColor=white&color=c59adc" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/lhhhappy/Earthquake_predictor.git?style=default&color=c59adc" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/lhhhappy/Earthquake_predictor.git?style=default&color=c59adc" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>❯ REPLACE-ME</code>

---

##  Features

<code>❯ REPLACE-ME</code>

---

##  Project Structure

```sh
└── Earthquake_predictor.git/
    ├── README.md
    ├── Result
    │   ├── Finetune_ES_net_mixer_california
    │   ├── Pretrain_ES_net_mixer_california
    │   └── Scratch_ES_net_mixer_california
    ├── __pycache__
    │   └── inference_utils.cpython-39.pyc
    ├── data_preprocess_pipeline
    │   ├── __pycache__
    │   ├── pipeline.py
    │   ├── station_dict_all.pkl
    │   └── utils.py
    ├── dataset
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── dataset_test.ipynb
    │   └── dataset_utils.py
    ├── experiment
    │   ├── data_preprocess_ipynb
    │   └── data_preprocess_py
    ├── inference.ipynb
    ├── inference_utils.py
    ├── last.ckpt
    ├── loss
    │   ├── __init__.py
    │   ├── __pycache__
    │   ├── loss_test.ipynb
    │   └── loss_utils.py
    ├── model
    │   ├── ES_net.py
    │   ├── ES_net_mixer.py
    │   ├── Earthquake_net.ipynb
    │   ├── __init__.py
    │   └── __pycache__
    ├── model_params.json
    ├── reference_project
    │   └── timemixer.ipynb
    ├── task
    │   ├── finetune_es_net_mixer.sh
    │   ├── scratch_es_net_mixer.sh
    │   ├── train_es_net.sh
    │   └── train_es_net_mixer.sh
    └── train.py
```


###  Project Index
<details open>
	<summary><b><code>EARTHQUAKE_PREDICTOR.GIT/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/inference.ipynb'>inference.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/model_params.json'>model_params.json</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/inference_utils.py'>inference_utils.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/last.ckpt'>last.ckpt</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/train.py'>train.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- dataset Submodule -->
		<summary><b>dataset</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/dataset/dataset_test.ipynb'>dataset_test.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/dataset/dataset_utils.py'>dataset_utils.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- experiment Submodule -->
		<summary><b>experiment</b></summary>
		<blockquote>
			<details>
				<summary><b>data_preprocess_py</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_py/usgs_data_area_save.py'>usgs_data_area_save.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_py/down_earthquake_data.py'>down_earthquake_data.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_py/log_energy.py'>log_energy.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_py/get_aij.py'>get_aij.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_py/download_GNSS_data.py'>download_GNSS_data.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>data_preprocess_ipynb</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_ipynb/earthquake_usgs.ipynb'>earthquake_usgs.ipynb</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_ipynb/data_preposs_gnss.ipynb'>data_preposs_gnss.ipynb</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_ipynb/data_preposs_earthquake.ipynb'>data_preposs_earthquake.ipynb</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_ipynb/plot_GNSS.ipynb'>plot_GNSS.ipynb</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/experiment/data_preprocess_ipynb/dataset.ipynb'>dataset.ipynb</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- model Submodule -->
		<summary><b>model</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/model/Earthquake_net.ipynb'>Earthquake_net.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/model/ES_net_mixer.py'>ES_net_mixer.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/model/ES_net.py'>ES_net.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- reference_project Submodule -->
		<summary><b>reference_project</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/reference_project/timemixer.ipynb'>timemixer.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- Result Submodule -->
		<summary><b>Result</b></summary>
		<blockquote>
			<details>
				<summary><b>Scratch_ES_net_mixer_california</b></summary>
				<blockquote>
					<details>
						<summary><b>logs</b></summary>
						<blockquote>
							<details>
								<summary><b>lightning_logs</b></summary>
								<blockquote>
									<details>
										<summary><b>version_0</b></summary>
										<blockquote>
											<table>
											<tr>
												<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Scratch_ES_net_mixer_california/logs/lightning_logs/version_0/events.out.tfevents.1732334234.a800.51804.0'>events.out.tfevents.1732334234.a800.51804.0</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Scratch_ES_net_mixer_california/logs/lightning_logs/version_0/hparams.yaml'>hparams.yaml</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											</table>
										</blockquote>
									</details>
								</blockquote>
							</details>
						</blockquote>
					</details>
					<details>
						<summary><b>checkpoints</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Scratch_ES_net_mixer_california/checkpoints/Val-epoch=104-Aggregative_Score=0.00.ckpt'>Val-epoch=104-Aggregative_Score=0.00.ckpt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Scratch_ES_net_mixer_california/checkpoints/last.ckpt'>last.ckpt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>Hyperparameters</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Scratch_ES_net_mixer_california/Hyperparameters/model_params.json'>model_params.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Scratch_ES_net_mixer_california/Hyperparameters/scratch_es_net_mixer.sh'>scratch_es_net_mixer.sh</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>Finetune_ES_net_mixer_california</b></summary>
				<blockquote>
					<details>
						<summary><b>logs</b></summary>
						<blockquote>
							<details>
								<summary><b>lightning_logs</b></summary>
								<blockquote>
									<details>
										<summary><b>version_0</b></summary>
										<blockquote>
											<table>
											<tr>
												<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Finetune_ES_net_mixer_california/logs/lightning_logs/version_0/events.out.tfevents.1732353976.a800.69753.0'>events.out.tfevents.1732353976.a800.69753.0</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Finetune_ES_net_mixer_california/logs/lightning_logs/version_0/hparams.yaml'>hparams.yaml</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											</table>
										</blockquote>
									</details>
								</blockquote>
							</details>
						</blockquote>
					</details>
					<details>
						<summary><b>checkpoints</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Finetune_ES_net_mixer_california/checkpoints/last.ckpt'>last.ckpt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Finetune_ES_net_mixer_california/checkpoints/Val-epoch=06-TPR=0.03.ckpt'>Val-epoch=06-TPR=0.03.ckpt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>Hyperparameters</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Finetune_ES_net_mixer_california/Hyperparameters/model_params.json'>model_params.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Finetune_ES_net_mixer_california/Hyperparameters/finetune_es_net_mixer.sh'>finetune_es_net_mixer.sh</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>Pretrain_ES_net_mixer_california</b></summary>
				<blockquote>
					<details>
						<summary><b>logs</b></summary>
						<blockquote>
							<details>
								<summary><b>lightning_logs</b></summary>
								<blockquote>
									<details>
										<summary><b>version_0</b></summary>
										<blockquote>
											<table>
											<tr>
												<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Pretrain_ES_net_mixer_california/logs/lightning_logs/version_0/hparams.yaml'>hparams.yaml</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											<tr>
												<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Pretrain_ES_net_mixer_california/logs/lightning_logs/version_0/events.out.tfevents.1732275008.a800.115371.0'>events.out.tfevents.1732275008.a800.115371.0</a></b></td>
												<td><code>❯ REPLACE-ME</code></td>
											</tr>
											</table>
										</blockquote>
									</details>
								</blockquote>
							</details>
						</blockquote>
					</details>
					<details>
						<summary><b>checkpoints</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Pretrain_ES_net_mixer_california/checkpoints/last.ckpt'>last.ckpt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Pretrain_ES_net_mixer_california/checkpoints/Val-epoch=25-Aggregative_Score=0.35.ckpt'>Val-epoch=25-Aggregative_Score=0.35.ckpt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>Hyperparameters</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Pretrain_ES_net_mixer_california/Hyperparameters/model_params.json'>model_params.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/Result/Pretrain_ES_net_mixer_california/Hyperparameters/train_es_net_mixer.sh'>train_es_net_mixer.sh</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- task Submodule -->
		<summary><b>task</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/task/finetune_es_net_mixer.sh'>finetune_es_net_mixer.sh</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/task/train_es_net.sh'>train_es_net.sh</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/task/scratch_es_net_mixer.sh'>scratch_es_net_mixer.sh</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/task/train_es_net_mixer.sh'>train_es_net_mixer.sh</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- loss Submodule -->
		<summary><b>loss</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/loss/loss_utils.py'>loss_utils.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/loss/loss_test.ipynb'>loss_test.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- data_preprocess_pipeline Submodule -->
		<summary><b>data_preprocess_pipeline</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/data_preprocess_pipeline/utils.py'>utils.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/lhhhappy/Earthquake_predictor.git/blob/master/data_preprocess_pipeline/pipeline.py'>pipeline.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with Earthquake_predictor.git, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python


###  Installation

Install Earthquake_predictor.git using one of the following methods:

**Build from source:**

1. Clone the Earthquake_predictor.git repository:
```sh
❯ git clone https://github.com/lhhhappy/Earthquake_predictor.git
```

2. Navigate to the project directory:
```sh
❯ cd Earthquake_predictor.git
```

3. Install the project dependencies:

echo 'INSERT-INSTALL-COMMAND-HERE'



###  Usage
Run Earthquake_predictor.git using the following command:
echo 'INSERT-RUN-COMMAND-HERE'

###  Testing
Run the test suite using the following command:
echo 'INSERT-TEST-COMMAND-HERE'

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/lhhhappy/Earthquake_predictor.git/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/lhhhappy/Earthquake_predictor.git/issues)**: Submit bugs found or log feature requests for the `Earthquake_predictor.git` project.
- **💡 [Submit Pull Requests](https://github.com/lhhhappy/Earthquake_predictor.git/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

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

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---
