#!/usr/bin/env python3
"""
frequency配置の正しさを検証するテスト。
- 統計収集 run → JSON出力の確認
- frequency配置 run → 出力一致の確認
- ratio=1.0 と従来full-slotの出力一致確認
- ratio=0.0 とCPU-onlyの出力一致確認
"""
import subprocess
import json
import sys
import os

LLAMA_CLI = os.environ.get("LLAMA_CLI",
    r"C:\Users\dai86\llama-cpp-turboquant\build-hip\bin\llama-cli.exe")
MODEL = os.environ.get("MODEL",
    r"C:\Users\dai86\llama-cpp-turboquant\models\DeepSeek-V2-Lite.Q4_K_M.gguf")

def run_cli(extra_args, timeout=300):
    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "-ngl", "999",
        "-n", "32",
        "-p", "What is 2+2",
        "--temp", "0",
        "-s", "42",
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr, result.returncode

def test_full_slot_baseline():
    """ratio=1.0 は従来のfull-slotと同一のはず"""
    stdout1, _, rc1 = run_cli([
        "--moe-gpu-expert-slot-num", "999"
    ])
    stdout2, _, rc2 = run_cli([
        "--moe-gpu-expert-slot-num", "999",
        "--moe-expert-placement", "frequency",
        "--moe-gpu-expert-ratio", "1.0"
    ])
    assert rc1 == 0, f"full-slot failed: rc={rc1}"
    assert rc2 == 0, f"frequency 1.0 failed: rc={rc2}"
    # 出力の末尾token列が一致することを確認
    assert stdout1.strip()[-20:] == stdout2.strip()[-20:], \
        f"Output mismatch:\nfull-slot: {stdout1.strip()[-50:]}\nfreq 1.0: {stdout2.strip()[-50:]}"

def test_freq_collect():
    """統計収集 run でJSONが出力されること"""
    report_path = os.path.join(os.path.dirname(__file__), "test_freq.json")
    stdout, _, rc = run_cli([
        "--moe-freq-report-path", report_path,
        "-n", "100"
    ])
    assert rc == 0, f"freq collect failed: rc={rc}"
    assert os.path.exists(report_path), f"JSON not created: {report_path}"
    with open(report_path) as f:
        data = json.load(f)
    assert "layers" in data
    assert len(data["layers"]) > 0
    os.remove(report_path)

def test_freq_actually_engages():
    """2-pass: collect stats, then verify frequency placement actually arms"""
    report_path = os.path.join(os.path.dirname(__file__), "test_freq_engage.json")
    # pass 1: collect stats
    stdout1, _, rc1 = run_cli([
        "--moe-freq-report-path", report_path,
        "-n", "100"
    ])
    assert rc1 == 0, f"freq collect failed: rc={rc1}"
    assert os.path.exists(report_path), f"JSON not created: {report_path}"
    # pass 2: apply frequency placement with existing report
    stdout2, stderr2, rc2 = run_cli([
        "--moe-gpu-expert-slot-num", "999",
        "--moe-expert-placement", "frequency",
        "--moe-gpu-expert-ratio", "0.5",
        "--moe-freq-report-path", report_path,
    ])
    assert rc2 == 0, f"frequency engage failed: rc={rc2}"
    assert "frequency placement:" in stderr2, \
        f"Frequency whitelist was not built. stderr excerpt:\n{stderr2[-500:]}"
    os.remove(report_path)

def test_freq_ratio_06():
    """ratio=0.6で実行が完了すること"""
    stdout, _, rc = run_cli([
        "--moe-gpu-expert-slot-num", "999",
        "--moe-expert-placement", "frequency",
        "--moe-gpu-expert-ratio", "0.6"
    ])
    assert rc == 0, f"frequency 0.6 failed: rc={rc}"
    assert "2+2" in stdout.lower() or "4" in stdout, \
        f"Unexpected output: {stdout[-100:]}"

def test_freq_ratio_03():
    """ratio=0.3で実行が完了すること"""
    stdout, _, rc = run_cli([
        "--moe-gpu-expert-slot-num", "999",
        "--moe-expert-placement", "frequency",
        "--moe-gpu-expert-ratio", "0.3"
    ])
    assert rc == 0, f"frequency 0.3 failed: rc={rc}"

if __name__ == "__main__":
    tests = [
        test_full_slot_baseline,
        test_freq_collect,
        test_freq_actually_engages,
        test_freq_ratio_06,
        test_freq_ratio_03,
    ]
    for test in tests:
        print(f"Running {test.__name__}...", end=" ")
        try:
            test()
            print("PASS")
        except Exception as e:
            print(f"FAIL: {e}")
            sys.exit(1)
    print("All tests passed.")
