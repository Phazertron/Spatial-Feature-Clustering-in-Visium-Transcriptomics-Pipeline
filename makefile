ENV_NAME = spatial-clustering

.PHONY: setup activate run-notebook report clean help

help: ## Show available targets
	@echo "Targets:"
	@echo "  setup         Create conda environment from environment.yml"
	@echo "  run-notebook  Launch Jupyter Lab with PYTHONPATH set"
	@echo "  report        Generate HTML report for the latest session"
	@echo "  clean         Remove generated results (plots, metrics, logs)"
	@echo ""
	@echo "On Windows use the PowerShell script instead:"
	@echo "  .\run.ps1 setup"
	@echo "  .\run.ps1 notebook"
	@echo "  .\run.ps1 clean"

setup: ## Create conda environment
	conda env create -f environment.yml

activate: ## Print activation command (cannot activate from Make)
	@echo "Run: conda activate $(ENV_NAME)"

run-notebook: ## Launch Jupyter Lab
	PYTHONPATH=$(PWD) conda run -n $(ENV_NAME) jupyter lab --notebook-dir=$(PWD)

report: ## Generate HTML/PDF report for the latest session
	PYTHONPATH=$(PWD) conda run -n $(ENV_NAME) python scripts/generate_report.py --format html

clean: ## Remove generated results
	rm -rf results/plots/*
	rm -rf results/metrics/*
	rm -rf results/runs/*/report.html
	rm -rf results/runs/*/report.pdf
