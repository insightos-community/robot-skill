"""Pilot 启动的隔离 Python Worker。"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path
from threading import RLock, Thread
from typing import Any

from .models import SkillCancelled, SkillFailure, StopRequest
from .protocol import ConcurrentJsonRpcPeer, JsonRpcMessage
from .rpc_context import RpcSkillContext


def _load(reference: str) -> Any:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"非法 Python 入口: {reference}")
    return getattr(importlib.import_module(module_name), attribute)


class Worker:
    """一次只承载一个 Skill Execution，所有物理动作仍由 Pilot 执行。"""

    def __init__(self, skill_dir: Path, peer: ConcurrentJsonRpcPeer) -> None:
        self.skill_dir = skill_dir.resolve()
        self.peer = peer
        self.skill_name = ""
        self.skill_version = ""
        self.run_function: Any | None = None
        self.stop_function: Any | None = None
        self.input_model: Any | None = None
        self.controllers: dict[str, Any] = {}
        self.context: RpcSkillContext | None = None
        self._execution_lock = RLock()

    def serve(self) -> None:
        # SKILL.md 只由 Pilot 解析。Worker 先报告进程就绪，再接收 Pilot 已验证的入口。
        self.peer.send_notification("ready", {"skill_dir": str(self.skill_dir)})
        while True:
            message = self.peer.next_request()
            if message.method == "worker.initialize":
                self._initialize(message)
            elif message.method == "skill.validate_input":
                self._validate_input(message)
            elif message.method == "skill.run":
                Thread(target=self._run, args=(message,), daemon=True).start()
            elif message.method == "skill.stop":
                Thread(target=self._stop, args=(message,), daemon=True).start()
            elif message.method == "worker.shutdown":
                self.peer.send_result(message.request_id, {"status": "shutting_down"})
                return
            else:
                self.peer.send_error(message.request_id, -32601, f"未知 Worker 方法: {message.method}")

    def _initialize(self, message: JsonRpcMessage) -> None:
        """仅接受 Pilot 校验后的 Runtime 描述，并保证同一进程不能被换成另一 Skill。"""

        try:
            runtime = dict(message.params["runtime"])
            if int(runtime["api_version"]) != 1:
                raise ValueError("Worker 只支持 runtime api_version=1")
            name = str(message.params["name"])
            version = str(message.params["version"])
            if self.skill_name and (self.skill_name != name or self.skill_version != version):
                raise ValueError("已经初始化的 Worker 不能切换 Skill 或版本")
            run_function = _load(str(runtime["entrypoint"]))
            stop_function = _load(str(runtime["stop_entrypoint"]))
            input_model = _load(str(runtime["input_model"]))
            controllers = {
                str(controller_name): _load(str(reference))()
                for controller_name, reference in dict(runtime.get("controllers") or {}).items()
            }
            self.skill_name = name
            self.skill_version = version
            self.run_function = run_function
            self.stop_function = stop_function
            self.input_model = input_model
            self.controllers = controllers
            self.peer.send_result(
                message.request_id,
                {
                    "status": "initialized",
                    "name": name,
                    "version": version,
                    "input_schema": input_model.model_json_schema(),
                },
            )
        except BaseException as exc:
            self.peer.send_error(message.request_id, -32000, f"Worker 初始化失败: {exc}")

    def _validate_input(self, message: JsonRpcMessage) -> None:
        """使用Skill声明的Pydantic Model校验输入，不执行Stage或物理Action。"""

        if self.input_model is None:
            self.peer.send_error(message.request_id, -32000, "Worker 尚未由 Pilot 初始化")
            return
        try:
            value = self.input_model.model_validate(message.params["input"])
            self.peer.send_result(
                message.request_id,
                {"valid": True, "input": value.model_dump(mode="json")},
            )
        except BaseException as exc:
            self.peer.send_result(
                message.request_id,
                {"valid": False, "error": str(exc)},
            )

    def _run(self, message: JsonRpcMessage) -> None:
        if self.run_function is None or self.stop_function is None or self.input_model is None:
            self.peer.send_error(message.request_id, -32000, "Worker 尚未由 Pilot 初始化")
            return
        with self._execution_lock:
            if self.context is not None and self.context.status not in {"completed", "failed"}:
                self.peer.send_error(message.request_id, -32001, "Worker 已有活动 Execution")
                return
            params = message.params
            # Worker仍保留最终执行边界。Pilot预检用于在创建Server Execution前
            # 给Agent可纠正诊断，这里再次使用同一个Model，不复制字段规则。
            validated_input = self.input_model.model_validate(params["input"])
            self.context = RpcSkillContext(
                self.peer,
                execution_id=str(params["execution_id"]),
                robot_ref=str(params["robot_ref"]),
                skill_input=validated_input.model_dump(mode="json"),
                checkpoint=params.get("checkpoint"),
                controllers=self.controllers,
                log_fields=dict(params.get("log_fields") or {}),
                feedback_cursors=dict(params.get("feedback_cursors") or {}),
            )
        try:
            asyncio.run(self._run_until_terminal(self.context))
            result = {
                "status": self.context.status,
                "result": self.context.result.model_dump(mode="json") if self.context.result else None,
                "error": self.context.failure,
            }
            self.peer.send_result(message.request_id, result)
        except SkillCancelled as exc:
            self.peer.send_result(message.request_id, {"status": "stopping", "message": str(exc)})
        except SkillFailure as exc:
            self.peer.send_result(
                message.request_id,
                {"status": "failed", "error": {"code": exc.code, "message": exc.message}},
            )
        except BaseException as exc:
            self.peer.send_error(message.request_id, -32010, f"Skill Worker 异常: {exc}")
        # skill.run 的 failed 只说明业务执行失败，Pilot 仍可能需要 on_stop
        # 来确认物理保持。保留本次上下文直到 Pilot 清理 Worker 或启动下一次
        # 执行，避免失败刚返回就丢失停止入口；上下文不触发动作或自动重试。

    async def _run_until_terminal(self, context: RpcSkillContext) -> None:
        if self.run_function is None:
            raise RuntimeError("Worker 尚未初始化")
        for _ in range(64):
            if context.status in {"completed", "failed"}:
                return
            await self.run_function(context)
        raise RuntimeError("Skill 在 64 次 Stage 进入后仍未到达终态")

    def _stop(self, message: JsonRpcMessage) -> None:
        context = self.context
        if context is None or self.stop_function is None:
            self.peer.send_error(message.request_id, -32002, "Worker 没有活动 Execution")
            return
        try:
            context.request_stop()
            stop_request = StopRequest.model_validate(message.params["request"])
            outcome = asyncio.run(self.stop_function(context, stop_request))
            self.peer.send_notification("stop.outcome", outcome.model_dump(mode="json"))
            self.peer.send_result(message.request_id, outcome.model_dump(mode="json"))
        except BaseException as exc:
            self.peer.send_error(message.request_id, -32011, f"Skill 停止处理失败: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Semantic Robot Skill Worker")
    parser.add_argument("--skill-dir", required=True)
    args = parser.parse_args(argv)
    skill_dir = Path(args.skill_dir).resolve()
    if not (skill_dir / "SKILL.md").is_file():
        parser.error("skill-dir 中缺少 SKILL.md")
    sys.path.insert(0, str(skill_dir))

    protocol_stdout = sys.stdout
    peer = ConcurrentJsonRpcPeer(sys.stdin, protocol_stdout)
    # 从此以后第三方 print 只能进入 stderr，stdout 专用于 JSON-RPC。
    sys.stdout = sys.stderr
    try:
        Worker(skill_dir, peer).serve()
        return 0
    except (EOFError, KeyboardInterrupt):
        return 0
    except BaseException as exc:
        # stdout 只能承载协议消息；即使 Worker 自身启动失败也输出结构化 stderr。
        sys.stderr.write(
            json.dumps(
                {"level": "error", "component": "skill_worker", "message": str(exc)},
                ensure_ascii=False,
            )
            + "\n"
        )
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
