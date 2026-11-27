.DEFAULT_GOAL := help

# --- Setup ---
install:           ## Install runtime deps and editable tcrp package
	pip install -r requirements.txt
	pip install -e . --no-build-isolation --no-deps

install-dev:       ## Install dev + test deps
	pip install -r requirements-dev.txt
	pip install -e . --no-build-isolation --no-deps
	pre-commit install

install-training:  ## Install full training deps
	pip install -r requirements-training.txt
	pip install -e . --no-build-isolation --no-deps

# --- Training ---
train-tier1:       ## Train MobileNetV2 Tier1
	python -m scripts.train_tier1 --run-name Tier1_MobileNetV2

train-tier2:       ## Train EfficientNetB4 Tier2
	python -m scripts.train_tier2 --backbone efficientnet_b4 --run-name Tier2_EfficientNet

train-tier2-ark:   ## Train Ark+ Tier2
	python -m scripts.train_tier2 --backbone ark_plus --run-name Tier2_ArkPlus

# --- Evaluation ---
evaluate-nih:      ## Evaluate on NIH test set
	python -m scripts.evaluate_tiered --dataset nih --run-name FinalEval_NIH

evaluate-chexpert: ## Cross-dataset zero-shot on CheXpert
	python -m scripts.evaluate_chexpert --run-name FinalEval_CheXpert

ablation:          ## Run the ablation training/orchestration runs (A8/A9/A11/A12/A15, A13/A14)
	python -m scripts.run_ablation --start A1 --end A15

evaluate-ablations: ## Compute genuine A1-A12,A15 test metrics from trained weights (needs GPU + NIH data)
	python -m scripts.evaluate_ablation --all

ablation-table:    ## Compile outputs/results/ablation.json from the per-row evaluation JSONs
	python -m scripts.build_ablation_json

stats:             ## Run statistical tests
	python -m scripts.statistical_tests

benchmark:         ## Latency + memory + carbon
	python -m scripts.benchmark_latency

# --- Synthetic ---
generate-synthetic: ## Generate multi-batch FID-gated SD synthetic dataset
	python -m scripts.generate_synthetics --seed-scope all --batch-size 50 --num-variations 3 --fid-threshold 50.0 --resume

# --- Export ---
export-onnx:       ## ONNX export + INT8 quantization
	python -m scripts.export_onnx --model tier1
	python -m scripts.export_onnx --model tier2 --quantize int8

# --- Quality ---
test:              ## Run all tests
	pytest tests/ -v --cov=tcrp --cov-report=html

test-unit:         ## Unit tests only
	pytest tests/unit -v

test-integration:  ## Integration tests
	pytest tests/integration -v

lint:              ## Lint code
	ruff check src/tcrp/ tests/ scripts/
	mypy src/tcrp/

format:            ## Auto-format
	ruff format src/tcrp/ tests/ scripts/

check-imports:     ## Verify architectural boundaries
	lint-imports

check-lang:        ## Verify code comments are English-only
	python -m scripts.check_comment_language

# --- Paper / Figures ---
figures:           ## Generate all figures
	python -m scripts.build_thesis_figures

clean:             ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov

help:              ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install install-dev install-training train-tier1 train-tier2 train-tier2-ark \
        evaluate-nih evaluate-chexpert ablation evaluate-ablations ablation-table stats benchmark generate-synthetic \
        export-onnx test test-unit test-integration lint format check-imports check-lang \
        figures clean help
