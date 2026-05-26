.DEFAULT_GOAL := help

# --- Setup ---
install:           ## Install runtime deps
	pip install -r requirements.txt

install-dev:       ## Install dev + test deps
	pip install -r requirements-dev.txt
	pre-commit install

install-training:  ## Install full training deps
	pip install -r requirements-training.txt

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

ablation:          ## Run all ablations A1-A15
	python -m scripts.run_ablation --start A1 --end A15

stats:             ## Run statistical tests
	python -m scripts.statistical_tests

benchmark:         ## Latency + memory + carbon
	python -m scripts.benchmark_latency

# --- Synthetic ---
generate-synthetic: ## Generate SD synthetic batch
	python -m scripts.generate_synthetics --n 500 --version v2

# --- Export ---
export-onnx:       ## ONNX export + INT8 quantization
	python -m scripts.export_onnx --model tier1
	python -m scripts.export_onnx --model tier2 --quantize int8

# --- Web ---
serve-api:         ## Start FastAPI (dev)
	uvicorn web.backend.app:app --reload --host 0.0.0.0 --port 8000

serve-frontend:    ## Start React dev server
	cd web/frontend && npm run dev

serve:             ## Start full stack via docker-compose
	docker-compose up --build

serve-prod:        ## Start production stack
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

docker-run:        ## Run full stack in Docker (daemon mode)
	docker-compose up -d

docker-stop:       ## Stop Docker stack and containers
	docker-compose down

# --- Quality ---
test:              ## Run all tests
	pytest tests/ -v --cov=core --cov=application --cov-report=html

test-unit:         ## Unit tests only
	pytest tests/unit -v

test-integration:  ## Integration tests
	pytest tests/integration -v

lint:              ## Lint code
	ruff check core/ application/ infrastructure/ web/ tests/
	mypy core/ application/ infrastructure/

format:            ## Auto-format
	ruff format core/ application/ infrastructure/ web/ tests/

check-imports:     ## Verify architectural boundaries
	lint-imports

check-lang:        ## Verify code comments are English-only
	python -m scripts.check_comment_language

# --- Docker ---
docker-build:      ## Build all images
	docker-compose build

docker-push:       ## Push to registry
	docker-compose push

# --- Thesis ---
figures:           ## Generate all thesis figures
	python -m scripts.generate_report_figures --output thesis/figures

thesis-build:      ## Compile LaTeX
	cd thesis && latexmk -pdf -interaction=nonstopmode main.tex

clean:             ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov

help:              ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install install-dev install-training train-tier1 train-tier2 train-tier2-ark \
        evaluate-nih evaluate-chexpert ablation stats benchmark generate-synthetic \
        export-onnx serve-api serve-frontend serve serve-prod docker-run docker-stop test test-unit \
        test-integration lint format check-imports check-lang docker-build docker-push \
        figures thesis-build clean help
