# Unit Tests for Eviz

This directory contains unit tests for the Eviz data processing pipeline and related components.

## Test Structure

The tests are organized to mirror the structure of the main codebase:

```
tests/
├── conftest.py                  # Common fixtures for all tests
├── unit/                        # Unit tests
│   └── lib/
│       └── data/
│           ├── factory/         # Tests for factory components
│           │   ├── test_factory.py
│           │   └── test_registry.py
│           ├── pipeline/        # Tests for pipeline components
│           │   ├── test_pipeline.py
│           │   ├── test_reader.py
│           │   ├── test_processor.py
│           │   ├── test_transformer.py
│           │   └── test_integrator.py
│           └── sources/         # Tests for data sources
│               └── test_base.py
```

## Running Tests

To run all tests, use the following command from the project root:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/unit/lib/data/pipeline/test_pipeline.py
```

To run a specific test class:

```bash
pytest tests/unit/lib/data/pipeline/test_pipeline.py::TestDataPipeline
```

To run a specific test method:

```bash
pytest tests/unit/lib/data/pipeline/test_pipeline.py::TestDataPipeline::test_init
```

## Test Coverage

To run tests with coverage:

```bash
pytest --cov=eviz
```

To generate a coverage report:

```bash
pytest --cov=eviz --cov-report=html
```

This will create an HTML coverage report in the `htmlcov` directory.

## Adding New Tests

When adding new tests:

1. Follow the existing directory structure to place your test files.
2. Use the naming convention `test_*.py` for test files.
3. Use the naming convention `Test*` for test classes.
4. Use the naming convention `test_*` for test methods.
5. Use fixtures from `conftest.py` where appropriate.
6. Add docstrings to test classes and methods to explain what they test.

## Mocking

The tests use the `unittest.mock` module to mock dependencies. This allows us to test components in isolation without requiring actual data files or external services.

## Fixtures

Common test fixtures are defined in `conftest.py`. These include:

- `mock_data_source`: A mock data source with a test dataset
- `test_dataset`: A test dataset
- `pipeline`: A DataPipeline instance
- `reader`: A DataReader instance
- `processor`: A DataProcessor instance
- `transformer`: A DataTransformer instance
- `integrator`: A DataIntegrator instance
- `temp_netcdf_file`: A temporary NetCDF file for testing
