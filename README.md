# Semantic Robot Skills

[English](README.md) | [简体中文](README.zh-CN.md)

This repository contains the official `semantic_robot_skill_sdk`, an isolated Python
Worker, and three recoverable Robot Skills: `semantic-navigation`, `grasp-object`,
and `place-object`.

Each Skill keeps only `SKILL.md`, `scripts/`, `references/`, `tests/` (optional in
release packages), and `requirements.lock`. Pilot scans only the `SKILL.md` frontmatter;
the repository no longer maintains `robot-skill.yaml` or a standalone Skill Library.

A Skill is responsible for fixed Stages, expectation checks, local recovery, and typed
Agent requests. It can only emit declared Actions and cannot import the Robot SDK, ROS,
MuJoCo, Isaac, or model libraries. Pilot launches the isolated Worker with
`python -m semantic_robot_skill_sdk.worker --skill-dir <dir>`; stdin/stdout carry only
JSON-RPC 2.0, and images, point clouds, and video are passed only as Artifact references.

The current tests use `MockSkillContext`, covering stable Action keys, Feedback cursors,
checkpoints, local recovery, stopping, and the three-Skill depalletizing closed loop.
When the physical robot and simulation backends are not connected, Pilot/Ability fail
explicitly.

## Project Structure

- `semantic_robot_skill_sdk/`: Worker protocol, execution context, and SDK.
- `semantic_robot_skills/skills/`: the three Skill packages for navigation, grasping, and placing.
- `tests/`: Worker / SDK tests; each Skill also has its own tests.

## Build and Installation

The Robot Skill Runtime SDK is installed as a Wheel into the Skill environment created
by Pilot. The three concrete Skills are packaged separately as Zips containing `SKILL.md`
and scripts, and are stored and distributed by the Semantic Server Robot Skill Registry;
they are not preinstalled with the R1 Pro robot type package.

Python **3.11+** and uv are required; quick-start uses **3.13**.

```bash
uv venv --python 3.13
uv pip install -e . pytest
PATH="$PWD/.venv/bin:$PATH" make test
uv build --wheel
```

You can also use the standard build commands:

```bash
python -m build --wheel
make test
```

The build artifact is named `semantic_robot_skill_sdk-<version>-py3-none-any.whl`, and
the Wheel contains only `semantic_robot_skill_sdk`. The three concrete Skills in this
repository are not included in this Wheel, to avoid bypassing the Server Registry's
installation, activation, and version selection.

The formal installation flow is:

```text
Server publishes the Skill package
→ select the Robot and the exact version
→ Pilot downloads to staging
→ validate SKILL.md, the entry point, required_actions, and requirements.lock
→ create an isolated environment and install the Skill SDK Wheel
→ atomically switch the active version
```

Robots of the same model can have different Skill versions installed. Upgrading the
Robot SDK, an Ability, or Pilot does not implicitly replace Robot Skills; a running
Worker always uses the Skill directory and environment pinned at startup.

## Local Debugging

The Worker entry point is only for Pilot or Runtime testing:

```bash
python -m semantic_robot_skill_sdk.worker --skill-dir \
  semantic_robot_skills/skills/grasp_object
```

stdin/stdout are dedicated to JSON-RPC. Running the Worker directly does not connect
to a Robot and cannot bypass Pilot to invoke an Ability. Full end-to-end validation
requires the Server to distribute the Skill and Pilot's Action routing to invoke the
exact Ability instance of the current Robot.

## FAQ

"Robot Skill not installed yet" means the registry is missing the specified package or
version. Building only the SDK Wheel or activating a Robot Bundle cannot fix this;
publish the exact requested Skill version and retry.

Start with Fake / simulation tests; once connected to hardware, Skills may produce real
motion and must only run in a controlled environment.

## Related Documents

[Detailed technical reference](README.reference.md) · [Skill engineering](semantic_robot_skills/skills/)

## License

Copyright 2026 InsightOS. First-party code is licensed under [Apache-2.0](LICENSE); for
third-party components and assets, see [NOTICE](NOTICE) and the
[license scope](LICENSE_SCOPE.md).

## Reproducible Builds on Three Platforms

See the [glibc, musl, and macOS build instructions](README.build.md): pinned source
versions, actual script entry points, tool requirements, local and CI commands, artifact
locations, and platform validation scope.
