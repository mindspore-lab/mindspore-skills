# New-Readiness-Agent 设计追踪文档

## 1. 文档目的

这份文档是 `new-readiness-agent` 的实现追踪记录，目标不是重复 `SKILL.md`
的对外说明，而是帮助后续修改代码时快速回答下面几个问题：

- 这个 skill 的真正职责是什么，哪些事情明确不做
- 从入口到最终 verdict 的状态流是什么
- 每个脚本文件各自负责什么
- 当前设计里哪些点是“故意这样做”，哪些点是历史问题修过来的
- 新需求进来时，应该优先改哪一层，避免继续把逻辑堆进主流程

建议每次修改 `new-readiness-agent` 之前，先通读本文档，再进入具体代码。

## 2. 产品定位

`new-readiness-agent` 是一个只读的、本地单机 NPU readiness certification
skill。它只回答：

- 当前 workspace 现在能不能开始训练或推理
- 当前仓库更像使用哪种 launcher / framework / Python env / CANN 路径
- 缺什么、卡在哪、哪些信息还需要用户确认

它明确不做：

- 修环境
- 安装依赖
- 下载模型或数据
- 修改代码、配置、shell 脚本
- 真正启动训练或推理
- 诊断运行后的 traceback、accuracy、performance 问题

换句话说，它是 “pre-run certification”，不是 “fixer”。

## 3. 总体执行流

公开入口只有一个：

- `scripts/run_new_readiness_pipeline.py`

整体执行顺序是：

1. `analyze_workspace`
2. `finalize_profile`
3. `build_confirmation_state`
4. `build_pending_validation` 或 `validate_profile`
5. `write_report_bundle`

对应的数据流可以概括为：

```text
CLI args
  -> scan (自动扫描候选)
  -> profile (把候选收敛成当前选择)
  -> confirmation_state (决定当前要问哪一个字段)
  -> validation (待确认态 or 最终校验态)
  -> verdict / lock / latest cache / report
```

最重要的状态切换只有两段：

- `phase=awaiting_confirmation`
  当前只输出一个待确认步骤，不做最终 READY/WARN/BLOCKED 判定
- `phase=validated`
  所有 gate 字段都已满足后，才进入最终校验并产出 READY/WARN/BLOCKED

## 4. 核心设计原则

### 4.1 单字段逐步确认

这个 skill 不再用“一张总表单统一确认”，而是一次只确认一个字段。原因是：

- 文本式宿主并不擅长处理大型动态表单
- 用户逐项确认更容易理解当前阻塞点
- 每次 rerun 的意图更清楚，便于 attempt 复用和缓存刷新

### 4.2 运行时与控制进程分离

skill 自己的控制 Python 和被认证的运行时环境不是一回事。真正的探测必须尽量切换到
“被选中的 runtime env 的 Python 视角”。

### 4.3 资产不是“路径字段”，而是“满足方式”

`config/model/dataset/checkpoint` 不能只建模成字符串路径，否则：

- Hugging Face Hub 无法表达
- HF cache 无法表达
- 脚本运行时下载无法表达
- inline config 无法表达

所以现在资产统一按 `source_type + locator` 建模。

### 4.4 报告与缓存分层

- `workspace-readiness.lock.json` 是给下游 agent 读取的稳定合同
- `readiness-verdict.json` 是当前 run 的完整机器可读结果
- `report.md` / `report.json` 是人类与调试辅助

## 5. 关键功能如何实现

### 5.1 Workspace 扫描

主要在 `scripts/new_readiness_core.py` 的 `analyze_workspace()`。

它负责：

- 枚举 workspace 文件
- 发现 launcher 候选
- 发现 entry script / config 候选
- 推断 target / framework 候选
- 发现 CANN 候选
- 调用环境候选模块和资产发现模块

扫描优先依赖：

- 显式输入
- `launch_command`
- 根目录常见脚本
- `Makefile`
- `requirements*.txt`
- `environment.yml` / `conda.yaml`
- YAML / JSON config

### 5.2 Launcher 推断

主要由 `parse_command_candidate()` 和 `build_launcher_candidates()` 实现。

支持：

- `python`
- `bash`
- `torchrun`
- `accelerate`
- `deepspeed`
- `msrun`
- `llamafactory-cli`
- `make`

推断依据包括：

- 显式 `launch_command`
- wrapper script
- `Makefile`
- `entry_script`

### 5.3 Target / Framework 推断

主要由：

- `build_target_candidates()`
- `build_framework_candidates()`

推断依据包括：

- entry script 名称
- launch command
- imports
- 依赖文本
- 用户显式 hint

这里的一个重要设计是：

- 即使工作区证据很弱，也会补 catalog 候选
- 所以用户总能看到 `training/inference`、`mindspore/pta/mixed` 这类标准选项

### 5.4 Python / Environment 选择

主要在 `scripts/environment_selection.py`。

候选来源包括：

- 用户显式指定
- launch command 中显式指向的环境
- 当前激活环境
- workspace-local `.venv` / `venv`
- workspace 的 conda 线索
- system Python

目前环境选择优先级是：

1. launch-command environment
2. active virtual environment
3. workspace-local environment
4. system Python

后续又补了一层：

- 如果 workspace 有 `environment.yml` / `conda.yaml` 且本机 `conda` 可用，会通过
  `conda env list --json` 生成 conda 候选
- `.venv` 和 conda 线索会同时列出，不再只依赖“当前激活环境”

### 5.5 资产发现

资产相关的核心拆成了三层：

- `scripts/asset_schema.py`
- `scripts/asset_discovery.py`
- `scripts/asset_validation.py`

#### `asset_schema.py`

负责统一资产结构：

- requirement
- candidate
- selected asset
- candidate 去重与排序
- locator 摘要

#### `asset_discovery.py`

负责发现资产候选，支持：

- `local_path`
- `hf_hub`
- `hf_cache`
- `script_managed_remote`
- `inline_config`

识别方式包括：

- config 中的 path / repo_id
- entry script 中的 `load_dataset(...)`
- `from_pretrained(...)`
- `snapshot_download(...)`
- `TrainingArguments(...)`
- workspace 下的 Hugging Face cache 布局

新增的设计约束：

- entry script 资产识别不再只依赖单行正则。当前实现优先做 AST 级分析，
  支持“顶层字符串常量 + 调用参数回填”的常见脚本写法，例如
  `MODEL_REPO_ID = "..."; AutoModel.from_pretrained(MODEL_REPO_ID)`。
- 正则只作为 AST 解析失败时的 fallback，避免把
  `AutoTokenizer.from_pretrained("...")` 之类的字面量调用误判成 model 线索。
- 资产信号现在统一覆盖 `config / model / tokenizer / dataset / checkpoint`，
  其中 `config` 和 `checkpoint` 也允许从 entry script 中提取本地路径候选，
  不再只依赖 workspace 扫描或 config 文件扫描。
- `tokenizer_asset` 已进入与 `model_asset / dataset_asset` 同一条
  “发现 -> 确认 -> 校验 -> 报告” 链路；是否进入确认流，取决于脚本或配置中
  是否发现了 tokenizer 线索。

#### `asset_validation.py`

负责按选中的 source_type 做最终验证，而不是硬编码“必须本地路径”。

### 5.6 Ascend / CANN 发现

相关逻辑在：

- `scripts/runtime_env.py`
- `scripts/new_readiness_core.py`

`runtime_env.py` 负责：

- 搜索可能的 `set_env.sh`
- 搜索 `version.cfg`
- 判断当前环境是否已激活 Ascend runtime
- 解析 `cann_path`
- 必要时 source `set_env.sh` 形成 probe 环境
- 先构造选中的 runtime env，再在其上叠加 sourced CANN env

`new_readiness_core.py` 负责把这些底层发现汇总成用户可理解的 check：

- `ascend-runtime`
- `cann-version`

这里后续做过一次重要收敛：

- 不再只报告 `CANN 8.5.0`
- 现在 summary 里会尽量带出路径来源，例如 `cann_path`、`set_env.sh`、
  `version.cfg`

### 5.7 Near-launch 验证

最终验证在 `validate_profile()`。

它会检查：

- target 是否已选
- launcher 是否已选
- runtime env 是否已选
- framework 是否已选
- Ascend / CANN 证据
- framework importability
- runtime dependencies
- launcher readiness
- entry script
- config/model/dataset/checkpoint 资产满足情况
- framework compatibility
- runtime-smoke

几个重要点：

- `probe_imports()` 现在走真实 `__import__`，不是 `find_spec`
- 这意味着 `torch_npu` 因 `libhccl.so` 缺失这类动态库错误会直接暴露出来
- 明确 `incompatible` 的 framework compatibility 现在会进入 blocker，而不是只有 warn
- PTA/NPU probe 不能只依赖“当前 shell 看起来已有 Ascend 变量”，而必须尽量在
  “selected runtime env + sourced CANN env” 组成的完整运行时上下文中执行

## 6. 当前状态模型

### 6.1 Phase

- `awaiting_confirmation`
- `validated`

### 6.2 Status

- `NEEDS_CONFIRMATION`
- `READY`
- `WARN`
- `BLOCKED`

### 6.3 Gate 字段

当前 readiness 真正依赖的核心 gate 包括：

- `target`
- `launcher`
- `framework`
- `runtime_environment`
- `entry_script`
- `config_asset`
- `model_asset`
- `dataset_asset`
- `cann_path`

注意：

- `launch_command` 已从强制确认序列中移除
- 现在默认由系统自动推导，而不是让用户最后再手输一遍

## 7. 文件职责总览

### 顶层文件

- `SKILL.md`
  对外工作流、能力边界、落盘策略说明
- `skill.yaml`
  manifest、输入声明、输出目录布局

### contract

- `contract/new-readiness-verdict.schema.json`
  verdict 的 JSON schema

### references

- `references/product-contract.md`
  产品职责与结果合同
- `references/decision-rules.md`
  推断和判定规则
- `references/cache-contract.md`
  latest cache 与 attempt 输出结构
- `references/design-trace.md`
  本文档，作为修改前的认知入口

### scripts

- `scripts/run_new_readiness_pipeline.py`
  唯一公共入口；解析参数、决定 `attempt_id`、选择输出目录、调用主流程和报告写入
- `scripts/new_readiness_core.py`
  主 orchestration；扫描、候选收敛、确认状态、最终校验
- `scripts/new_readiness_report.py`
  verdict、lock、latest cache、report bundle 的写入逻辑
- `scripts/environment_selection.py`
  Python / virtualenv / uv / conda / system Python 候选发现与排序
- `scripts/asset_schema.py`
  资产模型定义与辅助函数
- `scripts/asset_discovery.py`
  config/model/dataset/checkpoint 资产候选发现
- `scripts/asset_validation.py`
  资产最终校验
- `scripts/runtime_env.py`
  Ascend runtime / CANN 检测与 probe 环境构建
- `scripts/ascend_compat.py`
  MindSpore / PTA 与 CANN / Python / package version 的兼容性判断

### tests

- `tests/test_run_new_readiness_pipeline.py`
  主流程与回归测试
- `tests/test_environment_selection.py`
  环境候选相关回归
- `tests/test_python37_compatibility.py`
  Python 3.7 类型标注兼容性回归
- `tests/test_report_schema.py`
  verdict schema 基础校验
- `tests/test_manifest_contract.py`
  skill manifest 合同校验
- `tests/test_skill_structure.py`
  结构完整性校验
- `tests/conftest.py`
  测试夹具与 fake python

## 8. 重要历史问题、原因与修正策略

下面这些不是“偶然 bug”，而是设计层面曾经踩过的坑。后续修改时，优先检查是否会把这些问题带回来。

### 8.1 在用户确认前就直接 BLOCKED

问题：

- 初版会在 `target/launcher/framework` 未确认时直接给 `BLOCKED`

原因：

- 校验阶段把“待确认信息”过早当成“硬失败”

修正策略：

- 先进入 `NEEDS_CONFIRMATION`
- 当前只暴露一个待确认步骤
- 只有确认链走完后才允许进入最终 `validate_profile()`

### 8.2 统一大表单不适合文本宿主

问题：

- 一次性展示整张 confirmation form，用户负担重，宿主也不稳定

原因：

- 当前宿主不是原生动态表单系统

修正策略：

- 改为单字段逐步确认
- 每一步只输出当前字段的编号选项

### 8.3 资产只按本地路径建模

问题：

- HF Hub / HF cache / script-managed remote / inline config 都无法表达

原因：

- 旧设计把 `model_path/dataset_path/config_path` 当成字符串路径

修正策略：

- 重构为 `source_type + locator`
- 资产发现与校验拆分成 schema / discovery / validation 三层

### 8.4 `launch_command` 过度确认

问题：

- 最后一步还要用户手动确认命令，实质上是在重复输入系统已知信息

原因：

- 把派生字段误建模成了强 gate

修正策略：

- 从确认序列里移除 `launch_command`
- 默认自动推导，只在未来真的出现冲突场景时再考虑单独确认

### 8.5 framework compatibility 只给模糊 warning

问题：

- 只提示“可能不匹配”，原因不清楚

原因：

- 兼容性细节只埋在 evidence 里，没有提升到 check summary

修正策略：

- check summary 里直接带 installed versions、recommended specs、reason
- 明确 `incompatible` 时进入 blocker

### 8.6 `torch_npu` 导入失败却只给 WARN

问题：

- `torch_npu` 实际运行不了，但 verdict 还是 `WARN`

原因：

- 旧实现只做 `find_spec` 式探测，抓不到动态库错误

修正策略：

- 改成真实 import probe
- 导入错误直接进入 `framework-importability` / `runtime-dependencies`
- `runtime-smoke` 跟随变成 `BLOCKED`

### 8.7 CANN 只有版本，没有位置

问题：

- 报告里只看得到 `8.5.0`，却不知道是从哪里来的

原因：

- 输出合同只保留版本，不保留路径与来源信息

修正策略：

- 在 summary、verdict、lock 里补 `cann_path`、`ascend_env_script_path`、
  `cann_version_file`、candidate paths

### 8.8 artifact 太多，且每一步都新建顶级目录

问题：

- 一次交互看上去生成了很多平级 run 目录

原因：

- 每次 `--confirm` rerun 都生成新的 `run_id`
- 旧目录布局基于 `runs/<run_id>/out/`

修正策略：

- 默认输出迁移到 `readiness-output/`
- 引入 `attempt_id`
- 同一轮逐步确认复用一个 attempt 目录
- `awaiting_confirmation` 只写轻量文件
- `validated` 才写完整 bundle

### 8.9 Python 3.7 类型标注崩溃

问题：

- 宿主如果用 Python 3.7，会在导入时因 `list[str]` / `re.Pattern[str]` 直接报错

原因：

- 实现误用了 3.9+ 风格的运行时可求值类型注解

修正策略：

- 回退到 `typing.List` / `typing.Pattern`
- 用单独测试做回归保护

### 8.10 conda 环境不出现在候选里

问题：

- 即使 workspace 明显有 conda 线索，列表里也只显示 `.venv` 和 system Python

原因：

- 旧实现只认“当前激活 conda”或“launch command 显式 conda”

修正策略：

- 在 workspace 有 conda spec 且本机 `conda` 可用时，调用
  `conda env list --json`
- 把 conda 与 `.venv` 同时列出

### 8.11 probe 没有真正进入 selected env + CANN env

问题：

- `.venv` 和显式 `cann_path` 都已识别，但 `torch/torch_npu` probe 仍然因为
  `libhccl.so` 缺失失败

原因：

- 旧实现一旦发现“当前环境里已有一点 Ascend 痕迹”，就可能直接停在
  `current_environment`
- 即使用户显式提供了 `cann_path`，也不一定会真正 source 该路径下的
  `set_env.sh`
- probe 进程还可能只拿到 `selected_python`，却没有把 `selected_env_root`
  对应的虚拟环境上下文一起带上

修正策略：

- 显式 `cann_path` 要优先于“当前环境似乎已激活”的弱证据
- 先构造 selected runtime env，再在其上 source `set_env.sh`
- import probe、package version probe、framework compatibility 都要复用同一套
  完整 runtime context
- Linux-specific 的 `set_env.sh` source 回归要保留最小测试保护；Windows 本地
  开发环境可以 skip 这类 case，但不能改变 Linux 语义

## 9. 当前仍需注意的脆弱点

这些点还没有完全消失，后续修改要格外小心：

- `new_readiness_core.py` 仍然偏大，后续再加逻辑时优先往辅助模块下沉
- script 级 Hugging Face 识别还是偏启发式，复杂封装调用仍可能漏检
- 宿主最终显示给用户的内容可能是对 verdict 的二次总结，不一定逐字反映 `report.md`
- `attempt_id` 复用依赖 latest cache 的 pending 状态；如果外部误删 latest cache，可能会回退成新 attempt
- 涉及 `set_env.sh`、`bash -lc`、`LD_LIBRARY_PATH` 的逻辑在真实 Linux/NPU
  环境与本地 Windows 开发机上的行为差异很大，跨平台测试要保持最小但关键的回归覆盖

## 10. 后续修改建议

### 10.1 修改前 checklist

每次动这个 skill 前，先确认：

- 这次改动属于扫描、确认、验证、还是报告层
- 是否会改变 `workspace-readiness.lock.json` 的稳定合同
- 是否会把派生信息重新变成用户必须手填的字段
- 是否会让 `new_readiness_core.py` 继续膨胀
- 是否真的需要新测试，还是扩一条现有关键路径断言就够了

### 10.2 优先修改顺序

推荐顺序是：

1. 先改辅助模块
2. 再改 `new_readiness_core.py` 的调用点
3. 再改报告与 contract
4. 最后补最小必要回归测试

### 10.3 不建议的修改方式

避免：

- 在 `validate_profile()` 里继续叠分支修特殊 case
- 新增“老逻辑 + 新逻辑并存”的兼容层
- 把宿主文本展示问题误当成核心 verdict 结构问题来修
- 为单次展示需要扩张 lock schema

## 11. 一句话总结

`new-readiness-agent` 的本质是：

- 用 workspace 证据做只读推断
- 用逐项确认把不确定项收敛成明确运行画像
- 用 near-launch probe 判断这条运行画像是否真的可启动
- 用稳定 lock 和 latest cache 把这次判断复用给其他 agent

后续所有修改，最好都围绕这四件事展开，而不是把它重新做成一个环境修复器或大而全的交互系统。

## 12. 2026-04 重构补充

### 12.1 主流程继续收缩，避免 `new_readiness_core.py` 重新膨胀

这次重构的目标不是新增功能，而是把已经稳定的能力重新分层，减少主流程文件继续变成“巨石文件”的风险。

本次抽出的职责：

- `scripts/candidate_utils.py`
  - 放通用候选项排序、catalog 合并、路径判断等纯函数
- `scripts/confirmation_flow.py`
  - 放逐项确认状态机、选项编号、确认覆盖解析、asset/runtime env 选择逻辑
- `scripts/runtime_probes.py`
  - 放真实 import probe、版本 probe、launcher readiness、check summary 组装

这样 `new_readiness_core.py` 保留为 orchestration 层，只负责：

- 调用 workspace 扫描
- 汇总 profile
- 驱动 confirmation state
- 调用最终 validation
- 组织 verdict evidence

后续如果再改 `new_readiness_core.py`，优先先问自己：

- 这是候选项处理问题吗？
- 这是确认状态机问题吗？
- 这是 probe / check summary 问题吗？

如果答案是“是”，应优先往上述辅助模块下沉，而不是直接继续往 core 里堆分支。

### 12.2 测试按关注点拆分，而不是继续堆单文件回归

原先的 pipeline 测试文件同时覆盖：

- confirmation flow
- asset 场景
- artifact 落盘
- validation / runtime probe

这会导致：

- 文件过长
- 修改某一层逻辑时，很难快速定位要改哪一组测试
- fake python 夹具与 probe 实现细节耦合得过紧

现在测试按关注点拆分为：

- `tests/test_pipeline_confirmation_flow.py`
- `tests/test_pipeline_assets.py`
- `tests/test_pipeline_artifacts.py`
- `tests/test_pipeline_validation.py`
- `tests/helpers.py`

保留原则：

- 只保留能覆盖关键行为分层的用例
- 不保留“同一路径、只换一两个字面值”的堆叠 case
- 公共的 pipeline 调用与断言 helper 统一下沉到 `tests/helpers.py`

### 12.3 fake python 夹具不要再依赖 probe 代码正文

这次重构暴露出一个测试层的历史问题：

- 旧夹具通过匹配 `-c` 里是否包含某段 probe 代码文本，来判断当前是在跑 import probe 还是 version probe
- 一旦 probe transport 形状变化，例如 `exec(PROBE_CODE)`、整理常量、抽模块，测试就会误判

修正策略：

- fake python 夹具只依赖稳定协议：
  - `sys.argv[1] == "-c"`
  - `sys.argv[3]` 的 mode 值，例如 `import` / `package_versions` / `framework_smoke`
- 不再依赖 probe 代码正文里是否包含 `importlib.util` 等字符串

这条原则很重要：
测试应校验“协议是否稳定”，而不是绑定“当前实现长什么样”。

### 12.4 这次重构后仍需坚持的约束

- 不把 `launch_command` 再拉回强制确认链
- 不把资产模型退回成单一 `*_path`
- 不让 probe 再退回到只看 `find_spec` 的浅检查
- 不为了宿主展示层临时需要，扩张 lock schema
- 不为测试方便回退到把逻辑塞回 `new_readiness_core.py`

### 12.5 PTA probe 不能把 `torch` 和 `torch_npu` 绑在同一个导入进程里

后续真实测试又暴露出一个 PTA 特有问题：

- 某些 PyTorch 版本会在 `import torch` 时自动加载 `torch_npu` backend
- 如果 readiness probe 再在同一个子进程里继续 `import torch_npu`
- 就可能触发重复注册或 backend autoload 冲突，形成假阴性

这类报错看起来像：

- `duplicate triton TORCH_LIBRARY registration`
- `Failed to load the backend extension: torch_npu`

修正策略：

- `probe_imports()` 对包做逐个独立子进程探测
- PTA 包之间不共享同一个导入进程
- `probe_package_versions()` 优先读包元数据，再回退到 import

这样即使 import 失败，也尽量还能拿到已安装版本，避免 compatibility 退化成 “version unavailable”。

### 12.6 `__unknown__` 的产品语义更接近 “skip check for now”

逐项确认里最后一个选项以前显示为 `unknown / not sure`。从用户语义上看，这容易让人以为是在回答“我不知道”，而不是“我现在先跳过这项确认”。

因此当前 trace 约定：

- 内部值仍保留 `__unknown__`
- 但用户可见 label 改为 `skip check for now`

这样既不改内部协议，也更符合逐项 readiness confirmation 的真实行为。

### 12.7 显式传入 `set_env.sh` 路径时，不能只按目录搜索 CANN

又一次真实测试说明，PTA/NPU 路径里最先要保证的是：如果用户已经明确提供了 `.../set_env.sh`，probe 前必须真的把它识别成可 source 的 Ascend 脚本。

这里踩过的坑是：

- `normalize_cann_path()` 本身支持把 `set_env.sh` 文件路径归一成候选
- 但 `candidate_ascend_env_scripts()` 之前只把 `cann_path` 当成“搜索根”
- 后续又调用 `search_root_for_ascend_env_scripts()`，而这个函数要求 root 是目录
- 所以当 `cann_path` 直接是文件路径时，候选会被静默丢掉
- 最终 `resolve_runtime_environment()` 拿不到 `ascend_env_script_path`，就不会 source 该脚本

这会直接表现为：

- 没提前 source `set_env.sh` 时，`import torch` 触发 `torch_npu` backend autoload
- 因缺失 `libhccl.so` 等动态库而失败

修正策略：

- 对显式 `cann_path` 先做一轮直接候选收集
- 如果它本身就是 `set_env.sh`，优先把它纳入 `ascend_env_script_path`
- 不要把“显式文件路径”退化成“从该路径再按目录搜索”

这条优先级应始终高于“当前环境似乎已有 Ascend 痕迹”的弱证据。

### 12.8 `discover_asset_catalog()` 仍属于 analyzer，但必须按依赖分阶段执行

又一次真实测试暴露出一个依赖时机问题：

- `discover_asset_catalog()` 在架构归属上属于 workspace analyzer
- 但它生成的 `model/dataset/config/checkpoint` requirement 和 candidate，实际上依赖更上游的运行画像
- 尤其是：
  - `target` 会影响 `model/dataset` 是否 required
  - `entry_script` 会影响脚本内 `from_pretrained(...)` / `load_dataset(...)` / inline config 的识别结果
  - 若用户最终确认了本地 `config` 文件，也可能进一步补强 model/dataset hint

旧问题出在：

- `analyze_workspace()` 一开始就跑 `discover_asset_catalog()`
- 当时只能拿到原始 CLI args，拿不到后续逐项确认出来的 `target` / `entry_script`
- 结果 `active_confirmation_sequence()` 看见的 asset bundle 还是旧的
- `model_asset` / `dataset_asset` 会因为 `required=false` 且 `candidates=[]` 被静默跳过

这次的修正原则是：

- 不把 `discover_asset_catalog()` 从 analyzer 层移走
- 但把 analyzer 视为两段：
  - base scan：先扫 launcher / framework / entry / environment / cann 等基础候选
  - asset enrichment：在 `target`、`entry_script`（以及必要时本地 `config`）收敛后，再刷新 asset catalog

代码层面的落点：

- `analyze_workspace()` 保留一次初始 `discover_asset_catalog()`，用于早期 target inference 和基础资产线索
- `finalize_profile()` 在 base profile 收敛后，调用 `resolve_profile_assets()`
- `resolve_profile_assets()` 会基于已确认的 `target + entry_script` 重新跑 `discover_asset_catalog()`
- 如果最终选中的 `config_asset` 是本地文件，再做第二次 asset refresh

以后如果再改确认链，要记住这条依赖顺序：

`target / entry_script / local config`
-> refresh asset catalog
-> choose config/model/dataset/checkpoint
-> build confirmation state

不能再回退到“资产目录只在最早扫描时生成一次”的做法。

### 12.9 Shared heuristics and asset registry must stay centralized

This refactor addressed two growth risks that had started to spread across the
skill:

- path-vs-HF-repo heuristics had drifted into multiple files
- asset kind metadata had to be updated in several places whenever a new asset
  was introduced

The current constraints are:

- `candidate_utils.py` is the single source of truth for
  `looks_like_local_path()` and `looks_like_hf_repo_id()`
- `asset_registry.py` is the single source of truth for:
  - shared `ASSET_KINDS`
  - shared asset confirmation labels/prompts/manual hints
  - which assets participate in validation gating
  - shared `ENTRY_PATTERNS` / `CONFIG_SUFFIXES`

Operationally, this means:

- `asset_discovery.py` should import heuristics instead of re-implementing them
- `confirmation_flow.py` should derive asset confirmation steps from the
  registry instead of hard-coding one block per asset
- `new_readiness_core.py` and `new_readiness_report.py` should iterate over
  shared asset kinds instead of maintaining private tuples

If we later add `processor_asset`, `adapter_asset`, or any new asset type,
extend the registry first. Do not copy-paste another per-file asset list.

### 12.10 Same-source asset options must collapse into one candidate

Real entry scripts often expose the same asset through multiple signals:

- a top-level constant such as `MODEL_REPO_ID = "Qwen/Qwen3-0.6B"`
- one or more runtime callsites such as `from_pretrained(MODEL_REPO_ID)`

If readiness turns every signal into a separate option, the confirmation UI
starts showing duplicate entries like:

- `script-managed model: Qwen/Qwen3-0.6B`
- `script-managed model: Qwen/Qwen3-0.6B`

That is not useful user choice. The user should choose between asset sources,
not between multiple pieces of evidence for the same source.

Current constraint:

- candidate identity must be source-aware but evidence-agnostic where
  appropriate
- `script_managed_remote` candidates dedupe by semantic source
  (`repo_id + entry_script + split`), not by callsite
- `inline_config` candidates dedupe by script-level config source, not by
  `TrainingArguments` callsite
- evidence from multiple matching callsites should be merged into the single
  surviving candidate, not dropped entirely

Implementation note:

- this normalization belongs in `asset_schema.py`, not in each individual asset
  builder
- if we later add new source types that can produce multiple equivalent
  callsites, extend candidate identity normalization there first

### 12.11 Script-derived HF assets should surface as HF sources, not `script-managed` choices

Another UX issue showed up once tokenizer/model/dataset options started being
derived from entry-script AST signals:

- users were being asked to choose between `script-managed tokenizer`,
  `HF Hub tokenizer`, and `HF cache tokenizer`
- for direct `from_pretrained("repo")` and `load_dataset("repo")` flows,
  `script-managed` and `HF Hub` are not meaningfully different user choices

The important distinction for the user is the asset source that will satisfy
readiness:

- local path
- local HF cache
- remote HF Hub repo

`script-managed` is evidence provenance, not a separate satisfaction mode.

Current constraint:

- auto-detected script hints for `model`, `tokenizer`, and `dataset` should
  surface as `HF Hub` and optionally `HF cache` candidates
- they should not surface an extra `script_managed_remote` option when the
  actual runtime source is still the same HF repo
- script provenance must still remain visible through descriptions/evidence,
  for example:
  - detected from the entry script
  - runtime will load/download from this HF Hub repo
  - a matching local HF cache already exists

Implementation note:

- this normalization belongs in `asset_discovery.py` and
  `confirmation_flow.py`
- discovery chooses the real user-facing source candidates
- confirmation descriptions explain that those candidates were inferred from
  the script

### 12.12 Script-derived asset descriptions should use one explanation pattern

Once `config / model / tokenizer / dataset / checkpoint` all started flowing
through the same asset-confirmation path, another UX inconsistency showed up:

- HF-derived assets explained that they came from the entry script and what
  runtime would do next
- script-derived local config/checkpoint options still used generic
  "detected candidate" wording

That made different asset kinds feel like they were produced by different
systems even though they were all inferred from the same entry-script analysis.

Current constraint:

- script-derived asset descriptions should all follow the same template:
  - detected from the entry script
  - explain how runtime will use that source
  - append locator detail such as path, repo ID, or cache path
- the exact wording can vary by asset kind:
  - config: runtime uses this local config path
  - checkpoint: runtime references this path for resume/load behavior
  - HF Hub assets: runtime will load/download from this repo
  - HF cache assets: a matching local cache already exists

Implementation note:

- keep this description templating centralized in `confirmation_flow.py`
- do not spread per-asset wording into discovery or report generation
