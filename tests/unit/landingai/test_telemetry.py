from landingai.telemetry import get_runtime_environment_info


def test_get_environment_info():
    info = get_runtime_environment_info()
    assert info["lib_type"] == "pylib"
    assert info["runtime"] == "pytest"
    assert "lib_version" in info
    assert "python_version" in info
    assert "os" in info
