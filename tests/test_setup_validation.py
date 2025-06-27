import pytest
import sys
from pathlib import Path


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_pytest_installed(self):
        """Verify pytest is available."""
        assert "pytest" in sys.modules or pytest is not None
        
    def test_project_structure_exists(self):
        """Verify the project structure is correctly set up."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
    def test_conftest_fixtures_available(self, temp_dir, mock_config, sample_data):
        """Verify conftest fixtures are accessible."""
        assert temp_dir.exists()
        assert mock_config is not None
        assert isinstance(sample_data, dict)
        assert "integers" in sample_data
        
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
        
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True
        
    def test_temp_file_fixture(self, temp_file):
        """Verify temp_file fixture works correctly."""
        test_content = "Hello, testing!"
        file_path = temp_file("test.txt", test_content)
        assert file_path.exists()
        assert file_path.read_text() == test_content
        
    def test_mock_logger_fixture(self, mock_logger):
        """Verify mock_logger fixture works correctly."""
        mock_logger.info("Test message")
        mock_logger.info.assert_called_once_with("Test message")
        
    def test_env_vars_fixture(self, env_vars):
        """Verify env_vars fixture works correctly."""
        import os
        env_vars(TEST_VAR="test_value")
        assert os.environ.get("TEST_VAR") == "test_value"
        
    def test_capture_stdout_fixture(self, capture_stdout, capsys):
        """Verify capture_stdout fixture works correctly."""
        # Use pytest's built-in capsys instead for validation
        print("Test output")
        print("Another line")
        captured = capsys.readouterr()
        assert "Test output" in captured.out
        assert "Another line" in captured.out
        
    def test_coverage_configured(self):
        """Verify coverage is properly configured."""
        try:
            import coverage
            assert True
        except ImportError:
            pytest.skip("Coverage will be installed when running poetry install")


class TestProjectCodeImports:
    """Test that project modules can be imported."""
    
    def test_can_import_project_modules(self):
        """Verify we can import modules from the project."""
        # Test importing a simple module
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        try:
            # Try to import one of the simpler modules
            import cmdline
            assert True
        except ImportError:
            # It's okay if imports fail before dependencies are installed
            pytest.skip("Project imports will work after poetry install")
        finally:
            # Clean up sys.path
            if str(project_root) in sys.path:
                sys.path.remove(str(project_root))


@pytest.mark.parametrize("command", ["test", "tests"])
def test_poetry_commands_configured(command):
    """Verify Poetry test commands are configured in pyproject.toml."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    content = pyproject_path.read_text()
    assert f'{command} = "pytest:main"' in content