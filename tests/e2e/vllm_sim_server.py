import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest
import requests
from loguru import logger


class VllmSimServer:
    """
    [vLLM simulator](https://llm-d.ai/docs/architecture/Components/inf-simulator)
    A vLLM simulator wrapper for pytest.
    """

    def __init__(
        self,
        port: int,
        model: str,
        lora: Optional[list[str]] = None,
        mode: Optional[str] = None,
        echo: Optional[bool] = None,
        random: Optional[bool] = None,
        time_to_first_token: Optional[float] = None,
        inter_token_latency: Optional[float] = None,
        max_loras: Optional[int] = None,
        max_cpu_loras: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
    ):
        self.port = port
        self.model = model
        self.lora = lora
        self.mode = mode
        self.echo = echo
        self.random = random
        self.time_to_first_token = time_to_first_token
        self.inter_token_latency = inter_token_latency
        self.max_loras = max_loras
        self.max_cpu_loras = max_cpu_loras
        self.max_num_seqs = max_num_seqs
        self.server_url = f"http://127.0.0.1:{self.port}"
        self.health_url = f"{self.server_url}/health"
        self.app_script = "./bin/llm-d-inference-sim"
        self.process: Optional[subprocess.Popen] = None
        if not Path(self.app_script).exists():
            message = (
                "The vLLM simulator binary is required for E2E tests, but is missing.\n"
                "To build it and enable E2E tests, please run:\n"
                "docker build . -f tests/e2e/vllm-sim.Dockerfile -o type=local,dest=./"
            )
            logger.warning(message)
            pytest.skip("vLLM simlator binary missing", allow_module_level=True)

    def get_cli_parameters(self) -> list[str]:
        parameters = ["--port", f"{self.port}", "--model", self.model]
        if self.lora is not None:
            parameters.extend(["--lora", ",".join(self.lora)])
        if self.mode is not None:
            parameters.extend(["--mode", self.mode])
        if self.echo is not None:
            parameters.extend(["--echo"])
        if self.random is not None:
            parameters.extend(["--random"])
        if self.time_to_first_token is not None:
            parameters.extend(["--time-to-first-token", f"{self.time_to_first_token}"])
        if self.inter_token_latency is not None:
            parameters.extend(["--inter-token-latency", f"{self.inter_token_latency}"])
        if self.max_loras is not None:
            parameters.extend(["--max-loras", f"{self.max_loras}"])
        if self.max_cpu_loras is not None:
            parameters.extend(["--max-cpu-loras", f"{self.max_cpu_loras}"])
        if self.max_num_seqs is not None:
            parameters.extend(["--max-num-seqs", f"{self.max_num_seqs}"])
        return parameters

    def start(self):
        """
        Starts the server process and waits for it to become healthy.
        """

        logger.info(f"Starting server on {self.server_url} using {self.app_script}...")
        cli_parameters = self.get_cli_parameters()
        command = " ".join([self.app_script, *cli_parameters])
        logger.info(f"Server command: {command}")
        self.process = subprocess.Popen(  # noqa: S603
            [self.app_script, *cli_parameters],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Decode stdout/stderr as text
        )

        # Wait for the server to start and become healthy
        max_retries = 20
        retry_delay_sec = 0.5
        for i in range(max_retries):
            try:
                response = requests.get(self.health_url, timeout=1)
                if response.status_code == 200:
                    logger.info(f"Server started successfully at {self.server_url}")
                    return
                else:
                    logger.warning(f"Got response with status: {response.status_code}")
                    logger.warning(response.json())
            except requests.ConnectionError:
                logger.warning(f"Waiting for server... (attempt {i + 1}/{max_retries})")
                time.sleep(retry_delay_sec)
        # If the loop completes without breaking, the server didn't start
        stdout, stderr = self.process.communicate()
        logger.error(f"Server failed to start after {max_retries} retries.")
        logger.error(f"Server stdout:\n{stdout}")
        logger.error(f"Server stderr:\n{stderr}")
        self.stop()  # Attempt to clean up
        pytest.fail("Server did not start within the expected time.")

    def stop(self):
        """
        Stops the server process.
        """
        if self.process:
            logger.info(f"Stopping server on {self.server_url}...")
            self.process.terminate()  # Send SIGTERM
            try:
                self.process.wait(timeout=1)  # Wait for the process to terminate
                logger.info("Server stopped successfully.")
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing it...")
                self.process.kill()  # Send SIGKILL if it doesn't terminate
                self.process.wait()
            self.process = None  # Clear the process reference

    def get_url(self):
        """
        Returns the base URL of the running server.
        """
        return self.server_url
