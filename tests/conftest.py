import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file in the temporary directory."""
    def _temp_file(filename="test_file.txt", content=""):
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path
    return _temp_file


@pytest.fixture
def mock_config():
    """Provide a mock configuration object."""
    config = MagicMock()
    config.debug = False
    config.verbose = True
    config.timeout = 30
    config.max_retries = 3
    return config


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "integers": [1, 2, 3, 4, 5],
        "floats": [1.0, 2.5, 3.7, 4.2, 5.9],
        "strings": ["alpha", "beta", "gamma", "delta", "epsilon"],
        "mixed": [1, "two", 3.0, True, None],
        "nested": {
            "level1": {
                "level2": {
                    "data": "deep_value"
                }
            }
        }
    }


@pytest.fixture
def env_vars():
    """Temporarily set environment variables."""
    original_env = os.environ.copy()
    
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = str(value)
        return os.environ
    
    yield _set_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_file_system(monkeypatch):
    """Mock file system operations."""
    mock_open_func = Mock()
    mock_exists = Mock(return_value=True)
    mock_isfile = Mock(return_value=True)
    mock_isdir = Mock(return_value=False)
    
    monkeypatch.setattr("builtins.open", mock_open_func)
    monkeypatch.setattr("os.path.exists", mock_exists)
    monkeypatch.setattr("os.path.isfile", mock_isfile)
    monkeypatch.setattr("os.path.isdir", mock_isdir)
    
    return {
        "open": mock_open_func,
        "exists": mock_exists,
        "isfile": mock_isfile,
        "isdir": mock_isdir
    }


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset imports between tests to ensure clean state."""
    import sys
    modules_before = set(sys.modules.keys())
    yield
    modules_after = set(sys.modules.keys())
    for module in modules_after - modules_before:
        if module.startswith("test_"):
            del sys.modules[module]


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    
    def get_output():
        return captured.getvalue()
    
    return get_output


@pytest.fixture(scope="session")
def test_resources_dir():
    """Provide path to test resources directory."""
    resources_path = Path(__file__).parent / "resources"
    resources_path.mkdir(exist_ok=True)
    return resources_path


@pytest.fixture
def mock_requests(monkeypatch):
    """Mock requests library for HTTP testing."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = "Mock response text"
    
    mock_get = Mock(return_value=mock_response)
    mock_post = Mock(return_value=mock_response)
    
    mock_requests = Mock()
    mock_requests.get = mock_get
    mock_requests.post = mock_post
    mock_requests.Response = Mock
    
    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr("requests.post", mock_post)
    
    return mock_requests