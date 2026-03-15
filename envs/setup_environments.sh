#!/usr/bin/env bash
#===============================================================================
# DrugSol - Conda Environment Setup Script
#===============================================================================

set -euo pipefail

MODE="${1:-research}"
ENVS_DIR="${2:-./envs}"
CACHE_DIR="${ENVS_DIR}/conda_cache"

LOG_DIR="${ENVS_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/setup_environments_$(date +%Y%m%d_%H%M%S).log"

log_to_file() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "${LOG_FILE}"; }
log_info()  { log_to_file "[INFO] $1"; }
log_ok()    { log_to_file "[OK] $1"; }
log_warn()  { log_to_file "[WARN] $1"; }
log_error() { log_to_file "[ERROR] $1"; echo "[ERROR] $1" >&2; }
console_msg() { echo "$1"; }

console_msg "[DrugSol] Setting up environments (log: ${LOG_FILE})"
log_info "=========================================="
log_info "DrugSol Environment Setup"
log_info "=========================================="
log_info "Mode: ${MODE}"
log_info "Environments directory: ${ENVS_DIR}"

mkdir -p "${CACHE_DIR}"

#===============================================================================
# Environment Definitions
#===============================================================================

ENV_DATA="${CACHE_DIR}/drugsol-data"
ENV_DATA_PACKAGES=(
    "python=3.10"
    "libstdcxx-ng>=12"
    "setuptools"
    "numpy<2.0"
    "pandas=2.1"
    "pyarrow"
    "scikit-learn"
    "matplotlib"
    "seaborn"
    "requests"
    "rdkit=2023.09"
    "mkl"
    "mkl-service"
    "xtb"
    "tqdm"
    "scipy"
    "joblib"
)
ENV_DATA_PIP=(
    "mordredcommunity"
    "chembl_webresource_client"
)

# TRAIN: Python 3.8 + PyTorch 1.13 for chemprop 1.6.1
# Use blas=*=openblas to avoid MKL conflicts while allowing PyTorch's bundled MKL
ENV_TRAIN="${CACHE_DIR}/drugsol-train"
ENV_TRAIN_PACKAGES=(
    "python=3.8"
    "libstdcxx-ng>=12"
    "libgomp"
    "setuptools"
    "libblas=*=*openblas"
    "numpy<1.24"
    "pandas<2.0"
    "pyarrow"
    "scikit-learn<1.3"
    "optuna"
    "rdkit=2022.09"
    "tqdm"
    "scipy"
    "joblib"
    "matplotlib"
    "seaborn"
)
ENV_TRAIN_CHANNELS="-c conda-forge"
# Install PyTorch, xgboost, lightgbm, chemprop, catboost via pip for better compatibility
ENV_TRAIN_PIP=(
    "torch==1.13.1+cu117"
    "xgboost"
    "lightgbm"
    "catboost"
    "chemprop==1.6.1"
)

ENV_BERT="${CACHE_DIR}/drugsol-bert"
ENV_BERT_PACKAGES=(
    "python=3.10"
    "libstdcxx-ng>=12"
    "setuptools"
    "numpy<2.0"
    "pandas=2.1"
    "pyarrow"
    "tqdm"
    "rdkit=2023.09"
    "pytorch>=2.6"
    "cpuonly"
    "transformers"
    "accelerate"
    "safetensors"
    "tokenizers"
    "huggingface_hub"
    "joblib"
)
ENV_BERT_CHANNELS="-c pytorch -c conda-forge"

#===============================================================================
# Helper Functions
#===============================================================================

check_env_exists() {
    local env_path="$1"
    [ -d "${env_path}" ] && [ -f "${env_path}/bin/python" ]
}

verify_env() {
    local env_path="$1"
    local test_import="$2"
    
    if LD_LIBRARY_PATH="${env_path}/lib:${LD_LIBRARY_PATH:-}" \
       micromamba run -p "${env_path}" python -c "${test_import}" 2>>"${LOG_FILE}"; then
        return 0
    fi
    return 1
}

create_env() {
    local env_path="$1"
    local env_name="$2"
    local channels="$3"
    shift 3
    local packages=("$@")
    
    log_info "Creating environment: ${env_name}"
    log_info "Path: ${env_path}"
    
    [ -d "${env_path}" ] && rm -rf "${env_path}"
    
    log_info "Installing packages: ${packages[*]}"
    micromamba create -p "${env_path}" ${channels} "${packages[@]}" -y >> "${LOG_FILE}" 2>&1
    
    log_ok "Environment ${env_name} created successfully"
}

install_pip_packages() {
    local env_path="$1"
    shift
    local packages=("$@")
    
    log_info "Installing pip packages: ${packages[*]}"
    LD_LIBRARY_PATH="${env_path}/lib:${LD_LIBRARY_PATH:-}" \
    micromamba run -p "${env_path}" pip install --quiet "${packages[@]}" >> "${LOG_FILE}" 2>&1
    log_ok "Pip packages installed"
}

install_pip_with_index() {
    local env_path="$1"
    shift
    local packages=("$@")
    
    log_info "Installing pip packages with extra index: ${packages[*]}"
    LD_LIBRARY_PATH="${env_path}/lib:${LD_LIBRARY_PATH:-}" \
    micromamba run -p "${env_path}" pip install --quiet \
        --extra-index-url https://download.pytorch.org/whl/cu117 \
        "${packages[@]}" >> "${LOG_FILE}" 2>&1
    log_ok "Pip packages installed"
}

#===============================================================================
# Setup Functions
#===============================================================================

setup_data_env() {
    log_info "Setting up drugsol-data environment"
    console_msg "[DrugSol] Checking drugsol-data..."
    
    local test_import="import pandas, numpy, sklearn, rdkit, joblib; print('OK')"
    
    if check_env_exists "${ENV_DATA}" && verify_env "${ENV_DATA}" "${test_import}"; then
        log_ok "drugsol-data already exists and is valid"
        console_msg "[DrugSol] drugsol-data: OK (cached)"
        return 0
    fi
    
    console_msg "[DrugSol] drugsol-data: Creating... (this may take a few minutes)"
    create_env "${ENV_DATA}" "drugsol-data" "-c conda-forge" "${ENV_DATA_PACKAGES[@]}"
    
    if [ ${#ENV_DATA_PIP[@]} -gt 0 ]; then
        install_pip_packages "${ENV_DATA}" "${ENV_DATA_PIP[@]}"
    fi
    
    if verify_env "${ENV_DATA}" "${test_import}"; then
        log_ok "drugsol-data verified successfully"
        console_msg "[DrugSol] drugsol-data: OK"
    else
        log_error "drugsol-data verification failed!"
        console_msg "[DrugSol] drugsol-data: FAILED (check log)"
        return 1
    fi
}

setup_train_env() {
    log_info "Setting up drugsol-train environment"
    console_msg "[DrugSol] Checking drugsol-train..."
    
    local test_import="import torch; v=torch.__version__; assert v.startswith('1.'), f'Need PyTorch 1.x, got {v}'; import pandas, numpy, sklearn, xgboost, lightgbm, optuna, joblib; print('OK')"
    
    if check_env_exists "${ENV_TRAIN}" && verify_env "${ENV_TRAIN}" "${test_import}"; then
        log_ok "drugsol-train already exists and is valid"
        console_msg "[DrugSol] drugsol-train: OK (cached)"
        return 0
    fi
    
    console_msg "[DrugSol] drugsol-train: Creating... (this may take a few minutes)"
    create_env "${ENV_TRAIN}" "drugsol-train" "${ENV_TRAIN_CHANNELS}" "${ENV_TRAIN_PACKAGES[@]}"
    
    log_info "Installing PyTorch and ML packages via pip..."
    install_pip_with_index "${ENV_TRAIN}" "${ENV_TRAIN_PIP[@]}"
    
    if verify_env "${ENV_TRAIN}" "${test_import}"; then
        log_ok "drugsol-train verified successfully"
        console_msg "[DrugSol] drugsol-train: OK"
    else
        log_error "drugsol-train verification failed!"
        console_msg "[DrugSol] drugsol-train: FAILED (check log)"
        return 1
    fi
}

setup_bert_env() {
    log_info "Setting up drugsol-bert environment"
    console_msg "[DrugSol] Checking drugsol-bert..."
    
    local test_import="import pandas, numpy, torch, transformers; print('OK')"
    
    if check_env_exists "${ENV_BERT}" && verify_env "${ENV_BERT}" "${test_import}"; then
        log_ok "drugsol-bert already exists and is valid"
        console_msg "[DrugSol] drugsol-bert: OK (cached)"
        return 0
    fi
    
    console_msg "[DrugSol] drugsol-bert: Creating... (this may take a few minutes)"
    create_env "${ENV_BERT}" "drugsol-bert" "${ENV_BERT_CHANNELS}" "${ENV_BERT_PACKAGES[@]}"
    
    if verify_env "${ENV_BERT}" "${test_import}"; then
        log_ok "drugsol-bert verified successfully"
        console_msg "[DrugSol] drugsol-bert: OK"
    else
        log_error "drugsol-bert verification failed!"
        console_msg "[DrugSol] drugsol-bert: FAILED (check log)"
        return 1
    fi
}

#===============================================================================
# Main
#===============================================================================

main() {
    log_info "Starting environment setup"
    
    setup_data_env || exit 1
    
    if [ "${MODE}" == "research" ]; then
        setup_train_env || exit 1
        setup_bert_env || exit 1
    elif [ "${MODE}" == "execution" ]; then
        setup_train_env || exit 1
        [ "${CHEMBERTA_ENABLED:-true}" == "true" ] && setup_bert_env || true
    fi
    
    log_info "=========================================="
    log_ok "All environments ready!"
    log_info "=========================================="
    
    console_msg "[DrugSol] All environments ready!"
}

main