FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Activate environment for subsequent commands
SHELL ["conda", "run", "-n", "spatial-clustering", "/bin/bash", "-c"]

# Install additional Linux-compatible dependencies
RUN pip install scikit-misc

# Copy project files
COPY . .

# Expose Jupyter Lab port
EXPOSE 8888

# Default: launch Jupyter Lab (override with docker run ... python scripts/generate_report.py)
CMD ["conda", "run", "-n", "spatial-clustering", \
     "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--notebook-dir=/app"]
