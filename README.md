# Ground Deformation Susceptibility Analysis

A reproduction of the paper "Advanced susceptibility analysis of ground deformation disasters using large language models and machine learning: A Hangzhou City case study"

This project reproduces and analyzes the study from: https://doi.org/10.1371/journal.pone.0310724

```bibtex
@article{Yu_2024, 
  title={Advanced susceptibility analysis of ground deformation disasters using large language models and machine learning: A Hangzhou City case study}, 
  volume={19}, 
  ISSN={1932-6203}, 
  url={http://dx.doi.org/10.1371/journal.pone.0310724}, 
  DOI={10.1371/journal.pone.0310724}, 
  number={12}, 
  journal={PLOS ONE}, 
  publisher={Public Library of Science (PLoS)}, 
  author={Yu, Bofan and Xing, Huaixue and Ge, Weiya and Zhou, Liling and Yan, Jiaxing and Li, Yun-an}, 
  editor={Gul, Muhammet}, 
  year={2024}, 
  month=dec, 
  pages={e0310724} 
}
```

Original data of the study available here: https://figshare.com/articles/dataset/ML_and_LLM/25907179

```bibtex
@misc{Yu_2024_data,
  doi = {10.6084/M9.FIGSHARE.25907179.V2},
  url = {https://figshare.com/articles/dataset/ML_and_LLM/25907179/2},
  author = {YU, Bofan},
  keywords = {Geology not elsewhere classified},
  title = {ML and LLM},
  publisher = {figshare},
  year = {2024},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Features

- Ground collapse susceptibility analysis using machine learning
- Multiple model implementations (original and alternative approaches)
- Feature importance analysis
- LLM-based analysis with multiple models and temperature settings
- Comprehensive evaluation with ROC curves and metrics

## Requirements

- Python 3.8+
- Required libraries:
  - scikit-learn
  - pandas
  - matplotlib
  - joblib
  - numpy
  - OpenAI API (for LLM analysis)

Full list of libraries is here: `requirements.txt` and `pyproject.toml`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RomanKyrychenko/groud_collapse.git
```
2. Navigate to the project directory:
```bash
cd groud_collapse
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Add a `.env` file with your own `OPENAI_API_KEY`.

## Docker

To build and run the Docker container for this project, follow these steps:

1. Build the Docker image:
```bash
docker build -t ground_collapse_analysis .
```

2. Run the Docker container:

```bash
docker run --rm -v $(pwd)/output:/app/output ground_collapse_analysis
```

This will execute the analysis and save the results in the output directory.


## Usage

Run the main analysis script:
```bash
python main.py --input_file "input/ground collapse.xlsx" --input_prompt_pdf "input/prompt.pdf"
```

Additional options:

```bash
python main.py --help
```

## Key parameters

- `--input_file`: Path to input data file
- `--input_prompt_pdf`: PDF file with LLM prompt
- `--llm_repetitions`: Number of LLM experiment repetitions
- `--llm_models`: List of LLM models to use (e.g., gpt-3.5-turbo, gpt-4o)
- `--llm_temperatures`: List of temperature values for LLM experiments

## Project Structure

- `src/`: Source code for models and analysis
  - [`original_model.py`](src/original_model.py): Implementation of the original stacking model
  - [`alternative_model.py`](src/alternative_model.py): Alternative implementation with hyperparameter tuning
  - [`fake_model.py`](src/fake_model.py): Baseline model for memorizer
  - [`original_test.py`](src/original_test.py): Evaluation of the original model
  - [`alternative_test.py`](src/alternative_test.py): Evaluation of the alternative model
  - [`data_preproc.py`](src/data_preproc.py) & [`alternative_data_preproc.py`](src/alternative_data_preproc.py): Data preprocessing modules
  - [`llm.py`](src/llm.py) & [`llm_analysis.py`](src/llm_analysis.py): LLM-based analysis modules
- `input/`: Input data files, including test datasets and prompt files for LLM analysis
- `output/`: Directory containing generated model files, plots, and results
  - [`original_predicted_results.csv`](output/original_predicted_results.csv): CSV file with predictions from the original model
  - [`alternative_predicted_results.csv`](output/alternative_predicted_results.csv): CSV file with predictions from the alternative model
  - [`fake_model_results.csv`](output/fake_model_results.csv): CSV file with predictions from the baseline fake model
  - [`roc_curve_comparison.png`](output/roc_curve_comparison.png): PNG image comparing ROC curves of different models
  - [`llm_agg_results.tex`](output/llm_agg_results.tex): LaTeX file with aggregated results from LLM experiments
  - [`llm_ground_collapse_plot.png`](output/llm_ground_collapse_plot.png): PNG image showing ground collapse analysis using LLMs

## Models

The project implements several models:

- `StackingModel`: The original stacking classifier from the paper
- `AlternativeStackingModel`: Enhanced version with RandomForest and hyperparameter tuning
- `FakeModel`: Baseline model for comparison
- `LLM Analysis`: Evaluation of various LLMs on the task
