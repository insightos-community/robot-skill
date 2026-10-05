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

"""正式 Robot Skill 包结构和依赖边界测试。"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "semantic_robot_skills" / "skills"
FORBIDDEN_IMPORTS = {
    "r1pro_sdk",
    "rospy",
    "rclpy",
    "mujoco",
    "isaacsim",
    "omni",
    "torch",
    "transformers",
    "cv2",
}


def test_distribution_wheel_contains_only_the_runtime_sdk() -> None:
    """类型包只安装 Worker SDK，具体 Skill 必须由 Server Registry 单独下发。

    仓库仍同时维护三个 Skill 的源码和测试，但它们不能被打进共享 SDK Wheel；
    否则新版本 Skill 会绕过安装、启用和按 Robot 选择版本的流程。
    """

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "semantic-robot-skill-sdk"
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "semantic_robot_skill_sdk*"
    ]
    assert "package-data" not in project["tool"]["setuptools"]


def _frontmatter_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    return text.split("---", 2)[1]


def _value(frontmatter: str, field: str, indent: int = 0) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(field)}: [\"']?([^\n\"']+)", frontmatter, re.MULTILINE)
    assert match, field
    return match.group(1).strip()


def _module_path(skill_dir: Path, reference: str) -> Path:
    module, separator, _attribute = reference.partition(":")
    assert separator and module and _attribute, f"非法入口引用：{reference}"
    return skill_dir.joinpath(*module.split(".")).with_suffix(".py")


def test_skill_md_is_the_only_manifest_and_all_entries_exist() -> None:
    """Pilot 只需扫描 SKILL.md，不应再出现第二份清单。"""

    assert not list(SKILLS.rglob("robot-skill.yaml"))
    discovered: set[str] = set()
    for skill_doc in sorted(SKILLS.glob("*/SKILL.md")):
        skill_dir = skill_doc.parent
        frontmatter = _frontmatter_text(skill_doc)
        name = _value(frontmatter, "name")
        assert name not in discovered, f"重复 Skill 名称：{name}"
        discovered.add(name)
        for field in ("entrypoint", "stop_entrypoint", "input_model", "state_model", "result_model"):
            assert _module_path(skill_dir, _value(frontmatter, field, 2)).is_file(), field
        assert "required_actions:\n" in frontmatter
        assert "stop_actions:\n" in frontmatter
        assert "schema_version: 1" not in frontmatter
        assert "schema_version: 2" in frontmatter

    assert discovered, "至少应发现一个可加载 Skill 包"


def test_skill_scripts_do_not_cross_the_robot_or_model_boundary() -> None:
    """Skill 只能组合 Action，不能导入 Robot SDK、引擎、ROS 或模型库。"""

    violations: list[str] = []
    for source in sorted(SKILLS.glob("*/scripts/*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                roots = set()
            for root in roots & FORBIDDEN_IMPORTS:
                violations.append(f"{source.relative_to(ROOT)} 导入了 {root}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                violations.append(f"{source.relative_to(ROOT)} 使用了 print()")
    assert violations == []


def test_stage_rgb_helper_is_self_contained_and_declared_only_for_normal_actions() -> None:
    """三个 R1 拆码垛 ZIP 携带同一 helper，不需要升级 SDK，也不在停止入口拍照。"""

    helpers: list[bytes] = []
    for directory in ("semantic_navigation", "grasp_object", "place_object"):
        skill_dir = SKILLS / directory
        frontmatter = _frontmatter_text(skill_dir / "SKILL.md")
        normal, stop = frontmatter.split("stop_actions:", 1)
        assert "{ type: sensor.capture_rgbd, schema_version: 2 }" in normal
        assert "sensor.capture_rgbd" not in stop
        helpers.append((skill_dir / "scripts" / "stage_evidence.py").read_bytes())
        tree = ast.parse((skill_dir / "scripts" / "skill.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef) and node.name in {"on_stop", "_retreat"}:
                assert "capture_stage_rgb" not in ast.unparse(node)
    assert len(set(helpers)) == 1


def test_runtime_sdk_does_not_own_business_or_semantic_map_models() -> None:
    """公共 Worker SDK 只承载运行协议，业务结果由各 Robot Skill 自己定义。"""

    public_api = (ROOT / "semantic_robot_skill_sdk" / "__init__.py").read_text(
        encoding="utf-8"
    )
    runtime_models = (ROOT / "semantic_robot_skill_sdk" / "models.py").read_text(
        encoding="utf-8"
    )
    for business_type in (
        "HeldObjectState", "PlacementTarget", "PlacedObjectState",
        "MapEntityRef", "ResolvedEntity",
    ):
        assert business_type not in public_api
    assert "map_generation" not in runtime_models
    assert "map_revision" not in runtime_models
