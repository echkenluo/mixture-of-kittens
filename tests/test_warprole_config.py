import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
def test_config_probe(tmp_path):
    exe = tmp_path / "probe"
    subprocess.run(["g++", "-std=c++20", "-DMOK_WARPROLE_HOST_ONLY", "-I", str(ROOT),
                    str(ROOT / "tests/warprole_config_probe.cpp"), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    assert "warprole config probe OK" in out
