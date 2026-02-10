param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("setup", "activate", "notebook", "report", "clean")]
    [string]$command
)

$envName = "spatial-clustering"

switch ($command) {

    "setup" {
        Write-Host "Creating conda environment '$envName'..."
        conda env create -f environment.yml
        Write-Host "Environment created."
    }

    "activate" {
        Write-Host "Activating environment '$envName'..."
        conda activate $envName
    }

    "notebook" {
        Write-Host "Launching Jupyter Lab with project root in PYTHONPATH..."
        $projectRoot = (Get-Location).Path
        $env:PYTHONPATH = $projectRoot

        # Auto-create session if none exists
        Write-Host "Initializing session..."
        conda run -n $envName python -c "from src.utils.session import SessionManager; SessionManager.get_or_create_session(profile='default')"

        conda run -n $envName jupyter lab --notebook-dir="$projectRoot" --PreferredApp.default_notebook_dir="$projectRoot"
    }

    "report" {
        Write-Host "Generating HTML report for latest session..."
        $projectRoot = (Get-Location).Path
        $env:PYTHONPATH = $projectRoot
        conda run -n $envName python scripts/generate_report.py --format html
    }

    "clean" {
        Write-Host "Cleaning results folder..."
        Remove-Item -Recurse -Force results\plots\* -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force results\metrics\* -ErrorAction SilentlyContinue
        Write-Host "Cleanup complete."
    }
}