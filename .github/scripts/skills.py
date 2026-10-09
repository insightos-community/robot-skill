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

import re, zipfile
from pathlib import Path
out=Path('.output/payload/robot-skills');out.mkdir(parents=True)
for name,directory in {'grasp-object':'grasp_object','semantic-navigation':'semantic_navigation','place-object':'place_object'}.items():
 root=Path('semantic_robot_skills/skills')/directory
 version=re.search(r'(?m)^version:\s*([^#\s]+)',(root/'SKILL.md').read_text())[1].strip("'\"")
 with zipfile.ZipFile(out/f'{name}-{version}.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(root.rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.orig'):z.write(p,p.relative_to(root))
