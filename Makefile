.PHONY: lint format test run-service

lint:
	ruff check service/ scripts/ databricks/
	black --check service/ scripts/ databricks/

format:
	ruff check --fix service/ scripts/ databricks/
	black service/ scripts/ databricks/

test:
	pytest service/tests/ -v

run-service:
	cd service && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
