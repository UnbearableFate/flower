# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Execute and monitor a Flower run."""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from subprocess import Popen
from typing import Dict, List, Optional


@dataclass
class RunTracker:
    """Track a Flower run (composed of a run_id and the associated process)."""

    run_id: int
    proc: Popen  # type: ignore
    logs: List[str] = field(default_factory=list)


class Executor(ABC):
    """Execute and monitor a Flower run."""

    @abstractmethod
    def set_config(
        self,
        config: Dict[str, str],
    ) -> None:
        """Register provided config as class attributes.

        Parameters
        ----------
        config : Optional[Dict[str, str]]
            A dictionary for configuration values.
        """

    @abstractmethod
    def start_run(
        self,
        fab_file: bytes,
        override_config: Dict[str, str],
    ) -> Optional[RunTracker]:
        """Start a run using the given Flower FAB ID and version.

        This method creates a new run on the SuperLink, returns its run_id
        and also starts the run execution.

        Parameters
        ----------
        fab_file : bytes
            The Flower App Bundle file bytes.
        override_config: Dict[str, str]
            The config overrides dict sent by the user (using `flwr run`).

        Returns
        -------
        run_id : Optional[RunTracker]
            The run_id and the associated process of the run created by the SuperLink,
            or `None` if it fails.
        """
