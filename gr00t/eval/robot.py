# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Any, Dict

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.nvtx_range_logger import log_nvtx_range
from gr00t.eval.service import BaseInferenceClient, BaseInferenceServer
from gr00t.model.policy import BasePolicy


class RobotInferenceServer(BaseInferenceServer):
    """
    Server with three endpoints for real robot policies
    """

    def __init__(self, model, host: str = "*", port: int = 5555, api_token: str = None):
        super().__init__(host, port, api_token)
        first_call = [True]  # mutable so closure can set

        def get_action_with_nvtx_log(observations: Dict[str, Any]) -> Dict[str, Any]:
            start_ns = time.monotonic_ns()
            if first_call[0]:
                first_call[0] = False
                try:
                    steps = getattr(model, "denoising_steps", None)
                    if steps is not None:
                        log_nvtx_range(f"CONFIG_DENOISING_STEPS_{steps}")
                except Exception:
                    pass
            log_nvtx_range("POLICY_INFER_START")
            try:
                return model.get_action(observations)
            finally:
                end_ns = time.monotonic_ns()
                log_nvtx_range("POLICY_INFER_END")
                # 서버 프로세스가 본 inference 시간 (nsys/분석과 비교용)
                log_nvtx_range(f"INFER_DURATION_MS_{(end_ns - start_ns) * 1e-6:.2f}")

        self.register_endpoint("get_action", get_action_with_nvtx_log)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )

    @staticmethod
    def start_server(policy: BasePolicy, port: int, api_token: str = None):
        server = RobotInferenceServer(policy, port=port, api_token=api_token)
        server.run()


class RobotInferenceClient(BaseInferenceClient, BasePolicy):
    """
    Client for communicating with the RealRobotServer
    """

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None):
        super().__init__(host=host, port=port, api_token=api_token)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)
