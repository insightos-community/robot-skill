.PHONY: check test

check:
	python -m compileall -q semantic_robot_skill_sdk semantic_robot_skills tests

# 禁用开发机自动加载的 ROS pytest 插件，确保本地和 CI 使用同一隔离基线。
test: check
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:cacheprovider -q tests semantic_robot_skills/skills/*/tests
