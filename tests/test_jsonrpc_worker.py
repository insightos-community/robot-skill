# Copyright 2026 InsightOS
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""隔离 Worker 和 JSON-RPC 2.0 管道的跨进程测试。"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from semantic_robot_skill_sdk.protocol import JsonRpcProtocolError, LineJsonRpcPeer
from semantic_robot_skills.skills.semantic_navigation.scripts.models import (
    ResolvedNavigationTarget,
    SemanticNavigationInput,
)
from semantic_robot_skill_sdk import Pose3D


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "semantic_robot_skills" / "skills" / "semantic_navigation"


def test_protocol_rejects_stdout_pollution() -> None:
    """第三方 print 一旦进入协议流，Pilot 必须明确失败而不是猜测内容。"""

    from io import StringIO

    peer = LineJsonRpcPeer(StringIO("third party output\n"), StringIO())
    with pytest.raises(JsonRpcProtocolError, match="非 JSON"):
        peer.read()


def test_navigation_worker_completes_over_real_pipes() -> None:
    """通过真实子进程完成导航，证明 Skill 不依赖进程内 MockContext。"""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "semantic_robot_skill_sdk.worker",
            "--skill-dir",
            str(SKILL_DIR),
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    messages: queue.Queue[dict[str, object] | BaseException] = queue.Queue()

    def read_stdout() -> None:
        try:
            for line in process.stdout:
                messages.put(json.loads(line))
        except BaseException as exc:  # 测试需要把协议污染传回主线程断言。
            messages.put(exc)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()

    def receive() -> dict[str, object]:
        item = messages.get(timeout=5)
        if isinstance(item, BaseException):
            raise item
        return item

    def send(value: dict[str, object]) -> None:
        process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        process.stdin.flush()

    ready = receive()
    assert ready["method"] == "ready"
    send(
        {
            "jsonrpc": "2.0",
            "id": "initialize-1",
            "method": "worker.initialize",
            "params": {
                "name": "semantic-navigation",
                "version": "0.4.0",
                "runtime": {
                    "api_version": 1,
                    "entrypoint": "scripts.skill:run",
                    "stop_entrypoint": "scripts.skill:on_stop",
                    "input_model": "scripts.models:SemanticNavigationInput",
                    "controllers": {},
                },
            },
        }
    )
    initialized = receive()
    assert initialized["result"]["status"] == "initialized"
    assert initialized["result"]["input_schema"]["title"] == "SemanticNavigationInput"
    target = ResolvedNavigationTarget(
        target_ref="semantic://station/loading-a",
        pose=Pose3D(frame_id="map", position_m=(1.0, 2.0, 0.0), revision="target-r1"),
    )
    skill_input = SemanticNavigationInput(
        target=target, require_visual_confirmation=True
    )
    send(
        {
            "jsonrpc": "2.0",
            "id": "validate-1",
            "method": "skill.validate_input",
            "params": {"input": skill_input.model_dump(mode="json")},
        }
    )
    validated = receive()
    assert validated["result"]["valid"] is True
    send(
        {
            "jsonrpc": "2.0",
            "id": "run-1",
            "method": "skill.run",
            "params": {
                "execution_id": "execution-navigation-1",
                "robot_ref": "robot://r1pro/fake-1",
                "input": skill_input.model_dump(mode="json"),
                "checkpoint": None,
                "log_fields": {
                    "robot_id": "fake-1",
                    "project_id": "project-1",
                    "task_id": "task-1",
                    "subtask_id": "subtask-1",
                    "skill_execution_id": "execution-navigation-1",
                },
            },
        }
    )

    action_by_id: dict[str, str] = {}
    action_types: list[str] = []
    action_parameters: dict[str, dict[str, object]] = {}
    log_records: list[dict[str, object]] = []
    evidence_reports: list[dict[str, object]] = []
    capture_keys: list[str] = []
    checkpoint_stage: str | None = None
    terminal: dict[str, object] | None = None
    while terminal is None:
        message = receive()
        method = message.get("method")
        if method == "log":
            log_records.append(message["params"])
            continue
        if method == "checkpoint":
            checkpoint_stage = message["params"]["state"]["stage"]
            continue
        if method == "event.report":
            if message["params"]["event"] == "stage.evidence":
                evidence_reports.append(message["params"])
            continue
        if method == "complete":
            continue
        if method == "observation.latest":
            send({"jsonrpc": "2.0", "id": message["id"], "result": None})
            continue
        if method == "action.start":
            action = message["params"]["action"]
            action_id = f"action-{len(action_types) + 1}"
            action_by_id[action_id] = action["type"]
            action_types.append(action["type"])
            action_parameters[action["type"]] = action["parameters"]
            if action["type"] == "sensor.capture_rgbd":
                capture_key = message["params"]["key"]
                capture_keys.append(capture_key)
                assert json.loads(capture_key.removeprefix("stage-rgb:"))[:3] == [
                    "execution-navigation-1", "semantic-navigation", checkpoint_stage,
                ]
                assert action["parameters"] == {"sensor_ids": ["camera.rgb"]}
                assert action["timeout_seconds"] == 3
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"action_id": action_id},
                }
            )
            continue
        if method == "action.feedback":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"feedback": [], "terminal": True},
                }
            )
            continue
        if method == "action.result":
            action_type = action_by_id[message["params"]["action_id"]]
            if action_type == "sensor.capture_rgbd":
                send({
                    "jsonrpc": "2.0", "id": message["id"], "result": {
                        "status": "succeeded", "physical_effect": "none",
                        "observations": [{
                            "kind": "sensor.frame", "source": "robot-sdk://fake-1/sensor/camera.rgb",
                            "value": {"sensor_id": "camera.rgb", "media_type": "image/png"},
                            "evidence_refs": ["pilot-artifact://pilot-1/stage-rgb"],
                        }],
                    },
                })
                continue
            outputs = {
                "navigation.plan_route": {
                    "route_ref": "route://loading-a/1",
                    "resolved_target": target.model_copy(
                        update={
                            "pose": target.pose.model_copy(
                                update={"position_m": (1.1, 2.0, 0.0)}
                            )
                        }
                    ).model_dump(mode="json"),
                },
                "navigation.follow_route": {
                    "final_pose_ref": "pose://robot/final",
                    "distance_to_target_m": 0.1,
                },
                "navigation.verify_arrival": {
                    "verdict": "achieved",
                    "final_pose_ref": "pose://robot/final",
                    "distance_to_target_m": 0.1,
                },
            }
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "status": "succeeded",
                        "output": outputs[action_type],
                        "physical_effect": "confirmed",
                    },
                }
            )
            continue
        if message.get("id") == "run-1":
            terminal = message
            continue
        raise AssertionError(f"未处理的 Worker 消息：{message}")

    assert terminal["result"]["status"] == "completed"
    assert action_types == [
        "sensor.capture_rgbd",
        "navigation.plan_route",
        "sensor.capture_rgbd",
        "navigation.follow_route",
        "sensor.capture_rgbd",
        "navigation.verify_arrival",
        "sensor.capture_rgbd",
    ]
    assert len(capture_keys) == len(set(capture_keys)) == 4
    assert len(evidence_reports) == 4
    assert all(report["stage_status"] is None for report in evidence_reports)
    assert (
        action_parameters["navigation.verify_arrival"]["target"]["pose"]["position_m"][
            0
        ]
        == 1.1
    )
    assert log_records
    assert all(record["robot_id"] == "fake-1" for record in log_records)
    assert all(record["stage"] for record in log_records)

    send(
        {
            "jsonrpc": "2.0",
            "id": "shutdown-1",
            "method": "worker.shutdown",
            "params": {},
        }
    )
    assert receive()["result"]["status"] == "shutting_down"
    process.stdin.close()
    process.wait(timeout=5)
    assert process.returncode == 0, process.stderr.read()
