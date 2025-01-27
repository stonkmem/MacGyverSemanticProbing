<div align="left" style="position: relative;">

  
<img src="[https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg](https://github.com/user-attachments/assets/579944b7-7a06-4c3a-bf19-b44c48d70f39)" align="right" width="30%" style="margin: -20px 0 0 20px;">

![intellegence_12885303](https://github.com/user-attachments/assets/d5ce9f8d-3ac0-45ba-8d32-742290651ec5)
<h1>MacGyver Semantic Probing</h1>
<p align="left">
	<em><code>❯ Python</code></em>
</p>
<p align="left">
	<!-- local repository, no metadata badges. --></p>
<p align="left">
	<img src="https://img.shields.io/badge/HuggingFace-B41717.svg?style=for-the-badge&logo=HuggingFace&logoColor=white" alt="Hugging Face">
	<img src="https://img.shields.io/badge/PyTorch-FFC107.svg?style=for-the-badge&logo=pytorch&logoColor=black" alt="Pytorch">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=for-the-badge&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white" alt="OpenAI">
	<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=for-the-badge&logo=Pydantic&logoColor=white" alt="Pydantic">
</p>
</div>
<br clear="right">

##  Table of Contents

I. [ Overview](#-overview)
II. [ Features](#-features)
III. [ Project Structure](#-project-structure)
IV. [ Getting Started](#-getting-started)
V. [ Project Roadmap](#-project-roadmap)
VI. [ Contributing](#-contributing)
VII. [ License](#-license)
VIII. [ Acknowledgments](#-acknowledgments)

---

##  Overview

This is the code repository for the research project "Think Outside the Bot: Automating Evaluation of Creativity in LLMs for Physical Reasoning with Semantic Entropy and Efficient Multi-Agent Judge". 

---

##  Features

Contains code to run an automated benchmark on the MacGyver dataset. 

---

##  Project Structure

```sh
└── MacGyverSemanticProbing/
    ├── export_data.py
    ├── install_dependencies.sh
    ├── keys.py
    ├── LICENSE
    ├── llmaaj.py
    ├── README.md
    ├── readmeai-gemini-v1.md
    ├── readmeai-gemini.md
    ├── requirements.txt
    ├── script.bat
    ├── src
    │   ├── benchmark.py
    │   ├── dabertaMNLI.py
    │   ├── data.py
    │   ├── GPT_run_benchmark.py
    │   ├── helper_funcs.py
    │   ├── llama_funcs.py
    │   ├── Llama_run_benchmark.py
    │   ├── LLMevalframeworks.py
    │   ├── Mixtral_run_benchmark.py
    │   ├── openai_funcs.py
    │   ├── process_data.py
    │   ├── read_data.py
    │   └── vicuna_run_benchmark.py
    └── test_code
        ├── sample_query_Llama.py
        ├── sample_query_vicuna.py
        └── test_llama70b.py
```


###  Project Index
<details open>
	<summary><b><code>Benchmark</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/export_data.py'>export_data.py</a></b></td>
				<td>- `export_data.py` consolidates processed data from the `src.process_data` module<br>- It generates a JSON file containing various evaluation metrics, including  simplistic and complex scoring metrics, classification probabilities, and response lists<br>- The output filename is configurable via command-line arguments, allowing for flexibility in data storage<br>- The script's purpose is to provide a structured, readily accessible format for the project's analytical results.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/install_dependencies.sh'>install_dependencies.sh</a></b></td>
				<td>- The script automates the installation of project dependencies<br>- It manages environment variables, clones repositories, installs Python packages (including llama-cpp-python, transformers, and others) using pip, and verifies CUDA installation<br>- The process ensures the project's runtime environment is correctly configured for execution, leveraging both system and user-specified locations for caching and configuration files.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/keys.py'>keys.py</a></b></td>
				<td>- Keys.py establishes secure connections to external services<br>- It initializes OpenAI and Hugging Face API clients, providing authentication credentials for interaction with their respective platforms<br>- This facilitates access to large language models and other resources within the broader project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/llmaaj.py'>llmaaj.py</a></b></td>
				<td>- The `llmaaj.py` file acts as a setup and data preparation module within a larger project (likely involving large language models)<br>- It authenticates with the Hugging Face Hub, imports necessary libraries (including those for interacting with OpenAI and processing data), and prepares a Pandas DataFrame from external Excel files containing problem-solution pairs<br>- This prepared data, specifically a subset of efficient/inefficient/infeasible solutions, is then used as input for subsequent modules (the code snippet cuts off before showing the full usage, but it suggests further processing involving OpenAI's API for factuality checks)<br>- In essence, this file sets the stage for downstream tasks by handling authentication and data loading/preprocessing.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td>- `requirements.txt` specifies the project's dependencies<br>- It lists all external Python packages required for the application to function correctly, including libraries for natural language processing, machine learning, data manipulation, and web requests<br>- These packages enable the project's core functionalities.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/script.bat'>script.bat</a></b></td>
				<td>- The script automates the setup of a  machine learning environment<br>- It clones a specified Git repository, installs necessary Python packages including those for large language models and CUDA support, and verifies CUDA installation<br>- The process ensures the project's dependencies are correctly configured for execution, streamlining the development workflow.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- jobs Submodule -->
		<summary><b>jobs</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/jobs\llama_job.pbs'>llama_job.pbs</a></b></td>
				<td>- The `llama_job.pbs` script orchestrates a high-performance computing job<br>- It sets up the environment, installs dependencies, and executes a series of Python scripts for a Llama 3.1 language model benchmark<br>- These scripts handle data processing, model interaction, and result export, culminating in a comprehensive benchmark analysis<br>- The job leverages significant computational resources, including multiple CPUs and GPUs.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- src Submodule -->
		<summary><b>src</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\benchmark.py'>benchmark.py</a></b></td>
				<td>- The benchmark script facilitates multi-step problem-solving using various large language models (LLMs)<br>- It iteratively generates solutions for multiple problems, selecting the highest-probability step at each iteration<br>- The script supports different LLMs and incorporates a MacGyver-style problem-solving prompt,  recording probabilities and hidden states for analysis<br>- Results are stored for further evaluation.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\dabertaMNLI.py'>dabertaMNLI.py</a></b></td>
				<td>- The `dabertaMNLI.py` module provides natural language inference (NLI) capabilities<br>- It leverages a pre-trained DeBERTa model to classify the relationship between two text snippets (hypothesis and premise) as entailment, contradiction, or neutral<br>- The module offers functions to retrieve both the classification label and associated probability scores, facilitating NLI tasks within the broader project.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\data.py'>data.py</a></b></td>
				<td>- The `data.py` script preprocesses a dataset of problem-solution pairs<br>- It downloads data, formats it for a MacGyver-style problem-solving task,  creating prompts that challenge a model to generate creative, single-step solutions<br>- The script filters for solvable problems, shuffles the data, and prepares it for model training or evaluation within the larger project.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\GPT_run_benchmark.py'>GPT_run_benchmark.py</a></b></td>
				<td>- The `GPT_run_benchmark.py` file serves as a benchmark script within a larger project (likely involving AI problem-solving)<br>- It utilizes a large language model (LLM), likely via the `llama_funcs` module (indicated by the import statement), to generate sequential steps towards solving a problem presented as a prompt<br>- The script focuses on evaluating the LLM's ability to produce concise, creative, and effective solutions within a constrained number of steps<br>- The code's purpose is to test and measure the performance of this problem-solving approach.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\helper_funcs.py'>helper_funcs.py</a></b></td>
				<td>- The `src\helper_funcs.py` file provides a collection of utility functions used throughout the larger project<br>- These functions, drawing on other modules like `src.openai_funcs` and `src.data`,  facilitate tasks such as text generation (using models like GPT),  factuality assessment, and potentially entailment analysis<br>- The file also includes functions for evaluating model performance using metrics like ROC AUC and accuracy<br>- In essence, it acts as a central repository of reusable helper functions supporting the core functionalities of the project.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\llama_funcs.py'>llama_funcs.py</a></b></td>
				<td>- The `llama_funcs.py` file serves as the core logic for interacting with large language models (LLMs), likely within a larger application<br>- It imports necessary libraries for interacting with Hugging Face models (via the `transformers` library) and manages parameters such as temperature and top-p for controlling LLM generation<br>- The file appears to offer command-line argument parsing to customize these parameters, suggesting flexibility in how the LLMs are used within the broader project<br>- The use of environment variables (e.g., `HF_TOKEN`) indicates integration with a Hugging Face account for model access<br>- In short, this file acts as the interface between the application and the chosen LLMs, handling model selection, parameter configuration, and generation requests.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\Llama_run_benchmark.py'>Llama_run_benchmark.py</a></b></td>
				<td>- `Llama_run_benchmark.py` serves as a benchmark script within a larger project focused on problem-solving using a large language model (likely Llama)<br>- It utilizes functions from other modules (indicated by the imports) to generate and evaluate solutions to a problem, presented as a multi-step challenge to the model<br>- The script's core purpose is to test and measure the model's ability to devise efficient, feasible solutions step-by-step, mimicking a MacGyver-like approach<br>- The benchmark likely assesses the model's performance based on the number of steps required to reach a solution and the quality of each step generated.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\LLMevalframeworks.py'>LLMevalframeworks.py</a></b></td>
				<td>- The `LLMevalframeworks.py` file provides a testing framework for the OpenAI interaction component within a larger project<br>- It uses the `openai_funcs` module (presumably containing functions to interact with the OpenAI API) and a vector database (ChromaDB) along with sentence embeddings (SentenceTransformer) – though these latter two are not directly used in the shown code snippet<br>- The primary function, `test_openai()`, demonstrates a basic interaction with the OpenAI API, verifying a simple question-answering capability<br>- The inclusion of a safety definition string suggests a broader project focus on evaluating the safety of AI-generated responses, though the provided code snippet doesn't directly implement this aspect.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\Mixtral_run_benchmark.py'>Mixtral_run_benchmark.py</a></b></td>
				<td>- The script runs benchmarks on a MacGyver problem-solving model<br>- It iteratively generates multi-step solutions, using a large language model to propose each step<br>- The process involves selecting the most probable solution at each step and refining the prompt for subsequent steps<br>- The script manages multiple problems and steps, recording probabilities and intermediate results for analysis<br>- Output includes the generated solutions and associated probabilities.</td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\openai_funcs.py'>openai_funcs.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\process_data.py'>process_data.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\read_data.py'>read_data.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/src\vicuna_run_benchmark.py'>vicuna_run_benchmark.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- test_code Submodule -->
		<summary><b>test_code</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/test_code\sample_query_Llama.py'>sample_query_Llama.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/test_code\sample_query_vicuna.py'>sample_query_vicuna.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing/blob/master/test_code\test_llama70b.py'>test_llama70b.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with MacGyverSemanticProbing, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


###  Installation

Install MacGyverSemanticProbing using one of the following methods:

**Build from source:**

1. Clone the MacGyverSemanticProbing repository:
```sh
❯ git clone ../MacGyverSemanticProbing
```

2. Navigate to the project directory:
```sh
❯ cd MacGyverSemanticProbing
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```




###  Usage
Run the benchmark using the following command:

```sh
python export_data.py (model name) (json file name) (factuality judgement: chateval or llmjudge) (entailment model: gpt4 or deberta) (LLMjudge: true/false) (temperature of model) (number of questions to run benchmark on) (output_hiddenstates) (starting problem number)
```

Things to note:
1. LLMjudge should be set to false, as the feature is deprecated.
2. output_hiddenstates should be set to false, to prevent massive output file sizes.
3. Entailment model should be set to deberta, as GPT-4o entailment consumes a large amount of credits. 

---
##  Project Roadmap

TBC

---

##  Contributing

- **💬 [Join the Discussions](https://LOCAL/GitHub/MacGyverSemanticProbing/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/GitHub/MacGyverSemanticProbing/issues)**: Submit bugs found or log feature requests for the `MacGyverSemanticProbing` project.
- **💡 [Submit Pull Requests](https://LOCAL/GitHub/MacGyverSemanticProbing/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone C:\Users\ckcza\Documents\GitHub\MacGyverSemanticProbing
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
6. **Push to LOCAL**: Push the changes to your forked repository.
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
   <a href="https://LOCAL{/GitHub/MacGyverSemanticProbing/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=GitHub/MacGyverSemanticProbing">
   </a>
</p>
</details>

---

##  License

This project is protected under the MIT License. For more details, refer to the license file.

##  Acknowledgments

Main icon provided by Freepik. 

---

