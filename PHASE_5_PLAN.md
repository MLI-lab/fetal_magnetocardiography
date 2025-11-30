# Phase 5: Testing Infrastructure - Detailed Plan (REVISED)

## Overview
Set up testing infrastructure focused on **integration tests** and **examples** that verify scientific correctness and demonstrate real usage. Skip exhaustive unit tests in favor of practical validation.

## Philosophy
- **Integration tests** verify the science works end-to-end
- **Examples** serve as both documentation and functional tests
- Focus on reproducibility and real workflows
- Target: 40-50% coverage (quality over quantity)

## Step 1: Setup Testing Framework (30 min)
### 1.1 Add pytest to pyproject.toml
- Add pytest and pytest-cov to dev dependencies
- Add pytest configuration section

### 1.2 Create test infrastructure
- Create `tests/conftest.py` with shared fixtures
- Create `tests/data/` directory for test data
- Create subdirectories: `tests/unit/`, `tests/integration/`

### 1.3 Create test utilities
- Helper functions for generating synthetic test data
- Fixtures for common test configurations
- Mock measurement configs

---

## Step 2: Integration Tests (1 hour)

### 2.1 Configuration Tests (`tests/integration/test_config.py`)
**Priority: HIGH** - Verify Phase 4 dataclass configs work
- Test PipelineConfig creation with defaults
- Test `from_dict()` conversion (backward compatibility)
- Test `to_dict()` conversion
- Test nested dataclass instantiation
- Test config with numpy arrays
- Estimated: 5-6 tests

### 2.2 Pipeline Tests (`tests/integration/test_pipeline.py`)
**Priority: HIGH** - Core workflow
- Test Pipeline context manager (enter/exit)
- Test GPU cleanup (with mocking)
- Test Pipeline accepts both dict and PipelineConfig
- Test error handling in context
- Estimated: 4-5 tests

### 2.3 Synthetic Pipeline Test (`tests/integration/test_synthetic_workflow.py`)
**Priority: CRITICAL** - End-to-end scientific validation
- Create small synthetic fMCG data (2 dipoles, 10 seconds)
- Run full pipeline: load → preprocess → solve → detect heartbeats
- Verify output structure (m_hat, r_hat, peaks, HR)
- Check scientific validity (HR in expected range, positions converge)
- Verify reproducibility (deterministic results)
- Estimated: 2-3 comprehensive tests

---

## Step 3: Example Scripts/Notebooks (1 hour)

### 3.1 Basic Usage Example (`examples/basic_usage.py`)
**Purpose:** Show minimal working example
- Load synthetic data
- Create PipelineConfig using dataclass
- Run pipeline with context manager
- Display results
- Can be run as: `python examples/basic_usage.py`

### 3.2 Custom Configuration Example (`examples/custom_config.py`)
**Purpose:** Show how to customize configs
- Create custom SolverConfig, PreprocessingConfig
- Show different parameter combinations
- Demonstrate dict vs dataclass usage

### 3.3 Synthetic Data Demo Notebook (`examples/synthetic_data_demo.ipynb`)
**Purpose:** Interactive tutorial + validation
- Generate synthetic fMCG signals
- Step through pipeline stages
- Visualize intermediate results
- Verify scientific correctness
- Serves as both: tutorial + integration test

---

## Step 4: Test Data Generation (30 min)

### 4.1 Create synthetic test datasets
- Small synthetic fMCG signal (1 second, 2 dipoles)
- Test sensor configurations
- Test measurement configs
- Save as pickle/npz in `tests/data/`

### 4.2 Create test fixtures
- Sample configs for different scenarios
- Mock system/measurement configs
- Expected outputs for validation

---

## Step 5: Run Tests & Check Coverage (15 min)
- Run pytest with coverage
- Verify all integration tests pass
- Check coverage report (target: 40-50%)
- Document how to run tests in README

---

## Success Criteria
- [ ] ✅ pytest installed and configured
- [ ] ✅ 10-15 integration tests passing
- [ ] ✅ 3 example scripts/notebooks that run successfully
- [ ] ✅ End-to-end synthetic pipeline test passes
- [ ] ✅ Code coverage 40-50% for core modules
- [ ] ✅ Examples documented and runnable
- [ ] ✅ Tests pass in clean environment

---

## Estimated Total Time: 2-3 hours (reduced from 3-4)

## Priority Focus Areas (REVISED):
1. **CRITICAL**: End-to-end synthetic pipeline test (validates science)
2. **HIGH**: Config integration tests (validates Phase 4)
3. **HIGH**: Pipeline context manager tests (validates Phase 4)
4. **HIGH**: Example scripts/notebooks (documentation + functional tests)

## Deferred to Later:
- Unit tests for individual functions (can add incrementally if needed)
- Exhaustive edge case testing
- Performance benchmarks
- CI/CD automation
- Full Pipeline with real data (needs actual dataset access)

---

## Execution Order:
1. **Step 1**: Setup pytest (15 min)
2. **Step 2.1**: Config integration tests (20 min)
3. **Step 2.2**: Pipeline tests (20 min)
4. **Step 4**: Create synthetic test data (20 min)
5. **Step 2.3**: Synthetic workflow test (30 min)
6. **Step 3**: Example scripts/notebooks (1 hour)
7. **Step 5**: Run tests & coverage (15 min)

**Total: ~2.5 hours**
