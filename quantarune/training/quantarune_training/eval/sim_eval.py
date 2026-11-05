"""Automated simulation evaluation via aic_engine.

Launches the full evaluation stack (Zenoh router, Gazebo sim + engine,
policy node) and parses the scoring output. Does NOT reimplement scoring —
uses aic_engine + aic_scoring directly to match competition spec.
"""

import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from quantarune_training.eval.metrics import parse_scoring_results

logger = logging.getLogger(__name__)


class SimEvalError(RuntimeError):
    """Raised when simulation evaluation cannot proceed."""


def _check_prerequisites() -> list[str]:
    """Check that required tools are available. Returns list of issues."""
    issues = []

    # Check ros2
    if shutil.which("ros2") is None:
        issues.append("ros2 not found in PATH")

    # Check that aic_bringup launch file is findable
    if not issues:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "aic_bringup"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            issues.append("aic_bringup package not found (is the AIC workspace sourced?)")

    # Check that aic_model is findable
    if not issues:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "aic_model"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            issues.append("aic_model package not found")

    # Check for gz (Gazebo)
    if shutil.which("gz") is None and shutil.which("gazebo") is None:
        issues.append("Gazebo (gz) not found — simulation cannot run")

    return issues


def run_sim_eval(
    checkpoint_dir: str,
    policy_module: str = "aic_flow_policy.RunFlowPolicy",
    engine_config: str | None = None,
    results_dir: str | None = None,
    timeout: int = 300,
) -> dict[str, float]:
    """Run evaluation in simulation and return scored metrics.

    This automates the 3-terminal evaluation process:
    1. Zenoh router (rmw_zenohd)
    2. Gazebo sim + aic_engine
    3. Policy node (aic_model with our policy)

    Args:
        checkpoint_dir: Path to model checkpoint directory.
        policy_module: Python module path for the policy class.
        engine_config: Path to aic_engine trial config YAML.
        results_dir: Directory for scoring output.
        timeout: Maximum seconds to wait for evaluation.

    Returns:
        Dict of metric name -> value.

    Raises:
        SimEvalError: If prerequisites are not met.
    """
    # Pre-flight checks
    issues = _check_prerequisites()
    if issues:
        msg = "Sim eval prerequisites not met:\n" + "\n".join(f"  - {i}" for i in issues)
        raise SimEvalError(msg)

    if results_dir is None:
        results_dir = os.path.expanduser("~/aic_results")

    # Clean previous results
    results_path = Path(results_dir)
    scoring_file = results_path / "scoring.yaml"
    if scoring_file.exists():
        scoring_file.unlink()

    # Set environment variables
    env = os.environ.copy()
    env["AIC_RESULTS_DIR"] = results_dir
    env["AIC_FLOW_CHECKPOINT"] = str(checkpoint_dir)
    env["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"

    processes = []

    try:
        # 1. Launch Zenoh router
        logger.info("Starting Zenoh router...")
        zenoh_proc = subprocess.Popen(
            ["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(("zenoh", zenoh_proc))
        time.sleep(2)

        if zenoh_proc.poll() is not None:
            raise SimEvalError(f"Zenoh router exited immediately (code {zenoh_proc.returncode})")

        # 2. Launch simulation + engine
        logger.info("Starting simulation and engine...")
        sim_cmd = [
            "ros2", "launch", "aic_bringup", "aic_gz_bringup.launch.py",
            "start_aic_engine:=true",
            "shutdown_on_aic_engine_exit:=true",
            "ground_truth:=false",
        ]
        if engine_config:
            sim_cmd.append(f"aic_engine_config_file:={engine_config}")

        sim_proc = subprocess.Popen(
            sim_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("sim+engine", sim_proc))

        # Wait for sim to initialize, checking it hasn't crashed
        for i in range(20):
            time.sleep(1)
            if sim_proc.poll() is not None:
                stdout = sim_proc.stdout.read().decode() if sim_proc.stdout else ""
                raise SimEvalError(
                    f"Sim/engine exited during startup (code {sim_proc.returncode}). "
                    f"Output:\n{stdout[-2000:]}"
                )

        # 3. Launch policy node
        logger.info(f"Starting policy: {policy_module}")
        policy_proc = subprocess.Popen(
            [
                "ros2", "run", "aic_model", "aic_model",
                "--ros-args",
                "-p", "use_sim_time:=true",
                "-p", f"policy:={policy_module}",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes.append(("policy", policy_proc))

        # 4. Wait for engine to complete
        logger.info(f"Waiting for evaluation (timeout: {timeout}s)...")
        try:
            sim_proc.wait(timeout=timeout)
            logger.info(f"Sim/engine exited with code {sim_proc.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning("Evaluation timed out")

        # 5. Parse results
        logger.info("Parsing scoring results...")
        metrics = parse_scoring_results(results_dir)

        if metrics:
            logger.info(f"Evaluation complete — {len(metrics)} metrics collected")
        else:
            logger.warning("No scoring metrics found — check if aic_engine completed successfully")

        return metrics

    finally:
        # Cleanup all processes
        for name, proc in processes:
            if proc.poll() is None:
                logger.info(f"Stopping {name} (pid {proc.pid})...")
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        logger.info("Evaluation processes cleaned up")
