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

"""Worker 与 Pilot 之间的 JSON-RPC 2.0 行协议。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TextIO


class JsonRpcProtocolError(RuntimeError):
    """消息不是合法 JSON-RPC 或远端返回错误。"""


@dataclass(frozen=True)
class JsonRpcMessage:
    method: str | None
    params: dict[str, Any]
    request_id: str | int | None
    result: Any = None
    error: dict[str, Any] | None = None


class LineJsonRpcPeer:
    """以一行一个 JSON 对象收发消息；二进制数据只能使用 Artifact 引用。"""

    def __init__(self, reader: TextIO, writer: TextIO) -> None:
        self._reader = reader
        self._writer = writer
        self._next_id = 1

    def send_request(self, method: str, params: dict[str, Any]) -> str:
        request_id = str(self._next_id)
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        return request_id

    def send_notification(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def send_result(self, request_id: str | int, result: Any) -> None:
        self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def send_error(self, request_id: str | int | None, code: int, message: str) -> None:
        self._write(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
        )

    def read(self) -> JsonRpcMessage:
        line = self._reader.readline()
        if line == "":
            raise EOFError("JSON-RPC 输入已经关闭")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JsonRpcProtocolError("Worker stdin/stdout 出现非 JSON 内容") from exc
        if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
            raise JsonRpcProtocolError("消息缺少 jsonrpc=2.0")
        params = value.get("params") or {}
        if not isinstance(params, dict):
            raise JsonRpcProtocolError("params 必须是对象")
        return JsonRpcMessage(
            method=value.get("method"),
            params=params,
            request_id=value.get("id"),
            result=value.get("result"),
            error=value.get("error"),
        )

    def call(self, method: str, params: dict[str, Any]) -> Any:
        request_id = self.send_request(method, params)
        while True:
            message = self.read()
            if message.request_id != request_id or message.method is not None:
                raise JsonRpcProtocolError("等待响应时收到乱序或未处理消息")
            if message.error is not None:
                raise JsonRpcProtocolError(str(message.error.get("message", "远端调用失败")))
            return message.result

    def _write(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self._writer.write(encoded + "\n")
        self._writer.flush()

class ConcurrentJsonRpcPeer(LineJsonRpcPeer):
    """后台读取并按 request ID 分发，允许 Pilot 在 Skill 运行时发送 stop。"""

    def __init__(self, reader: TextIO, writer: TextIO) -> None:
        super().__init__(reader, writer)
        from queue import Queue
        from threading import Lock, Thread

        self._incoming: Queue[JsonRpcMessage | BaseException] = Queue()
        self._responses: dict[str, Queue[JsonRpcMessage | BaseException]] = {}
        self._responses_lock = Lock()
        self._writer_lock = Lock()
        self._reader_thread = Thread(target=self._read_forever, daemon=True)
        self._reader_thread.start()

    def call(self, method: str, params: dict[str, Any]) -> Any:
        from queue import Queue

        response_queue: Queue[JsonRpcMessage | BaseException] = Queue(maxsize=1)
        with self._responses_lock:
            request_id = str(self._next_id)
            self._next_id += 1
            self._responses[request_id] = response_queue
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        response = response_queue.get()
        if isinstance(response, BaseException):
            raise response
        if response.error is not None:
            raise JsonRpcProtocolError(str(response.error.get("message", "远端调用失败")))
        return response.result

    def next_request(self) -> JsonRpcMessage:
        message = self._incoming.get()
        if isinstance(message, BaseException):
            raise message
        return message

    def _write(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._writer_lock:
            self._writer.write(encoded + "\n")
            self._writer.flush()

    def _read_forever(self) -> None:
        try:
            while True:
                message = self.read()
                if message.method is not None:
                    self._incoming.put(message)
                    continue
                response_id = str(message.request_id)
                with self._responses_lock:
                    target = self._responses.pop(response_id, None)
                if target is None:
                    self._incoming.put(JsonRpcProtocolError(f"未知响应 ID: {response_id}"))
                else:
                    target.put(message)
        except BaseException as exc:
            self._incoming.put(exc)
            with self._responses_lock:
                pending = tuple(self._responses.values())
                self._responses.clear()
            for target in pending:
                target.put(exc)
