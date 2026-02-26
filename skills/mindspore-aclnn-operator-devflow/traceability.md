# Traceability（文档溯源映射）

目的：记录本 skill 的流程/清单来源于哪些原始文档，便于维护者核对"某条要求来自哪里"。

> **注意**：下表中的源文档（`算子流程/` 目录下的 18 篇）**不随 skill 分发**。
> 其内容已全部提炼到 `reference.md`、`checklists.md`、`examples.md` 中，
> skill 使用者无需拥有这些源文档即可正常使用。本文件仅供 skill 维护和溯源时参考。

## 源文档 → skill 落点

| 源文档（算子流程） | 主要贡献点 | skill 落点（文件 / 章节） |
| --- | --- | --- |
| `ACLNN_nsa_compress_适配开发经验.md` | 端到端落地顺序、踩坑与排障、Infer/PyBoost/KBK/bprop/测试经验 | `SKILL.md` Quick Start；`reference.md` 多处（推导策略、坑） |
| `MindSpore_开发者架构文档.md` | MindSpore 分层架构、模块定位 | `reference.md`（目录定位与背景认知） |
| `参考feature.md` | 以 `lightning_indexer` 为例的完整 RFC/实现/测试/文档/验收报告格式 | `reference.md`（验证项、用例覆盖、异常清单）；`examples.md` 触发样例 |
| `算子开发指导/1. 概述.md` | 对标策略（对齐 PyTorch/PTA）、性能与特性完备性原则 | `checklists.md` 0/6；`reference.md` 10/12/17 |
| `算子开发指导/2. 接口开发.md` | functional/nn/tensor 接口开发规范；functional 必用 `_get_cache_prim`；construct 校验用 `@constexpr`；Tensor GE 映射 | `reference.md` §19（接口开发要点）；`checklists.md` §3b（接口层） |
| `算子开发指导/3. 算子开发.md` | GeneralInfer 优先；动态 shape 算子分类；bprop expander 基本模板；need_compute_grad_out；InferValue 常量折叠 | `reference.md` 4/14/22/26；`checklists.md` 2/5 |
| `算子开发指导/4. 算子关键特性.md` | 动态 shape/rank 的 -1/-2 约定；动态 shape 三分类；`set_inputs` 自验；vmap 注册与测试要点 | `checklists.md` 6；`reference.md` §4.2/§23（vmap）/§27（动态 shape 分类） |
| `算子开发指导/5. 资料开发指导.md` | 中英文一致、接口列表字母序、文件名/标题/接口一致、示例可运行等文档规范 | `reference.md` 11；`checklists.md` 9 |
| `算子开发指导/7. 接口性能自验工具.md` | apitimewrapper 性能打点方法 | `reference.md` 12；`examples.md` 性能自验场景 |
| `算子开发指导/11.算子开发开源运作规范（试行）.md` | RFC 驱动流程与验收点（稳定性多次运行） | `reference.md` 13；`checklists.md` 6/0 |
| `ACLNN算子适配指导/Aclnn算子对接开发整体流程.md` | 方案设计、评审/交付、问题处理（CANN vs 框架）与书面结论 | `SKILL.md` Pre-A/Pre-B；`reference.md` 10/20；`checklists.md` 0 |
| `ACLNN算子适配指导/aclnn开发示例.md` | "三分类"决策；先自动生成再拷贝改造；KBK/PyBoost 典型路径；bprop I/O 规则；SetUnusedInputs | `reference.md` 2/6/7；`SKILL.md` Quick Start；`checklists.md` 1/5 |
| `ACLNN算子适配指导/算子反向注意事项.md` | OutZeros/ReturnZeros；ZerosLikeExt；inplace 反向 CloneInplaceInput；KBK 反向里 inplace 保序 Depend | `reference.md` 14；`checklists.md` 5 |
| `ACLNN算子适配指导/安全编码培训-算子代码检视.md` | 算子代码检视范围（正反向+Grad 单算子+多后端） | `reference.md` 15；`checklists.md` 10 |
| `ACLNN算子适配指导/1. ResizeKernelLaunch实现优化.md` | 禁止 Infer 改属性；Resize/Launch 分工；无意义输出忽略；compute-depend 输出 shape 更新；避免运行期内存申请 | `reference.md` 16；`checklists.md` 4 |
| `ACLNN算子适配指导/显存占用情况自验指导.md` | max memory 对齐 PTA 的方法 | `reference.md` §17.2；`checklists.md` 8 |
| `ACLNN算子适配指导/精度零偏差自验指导.md` | 0 偏差（bitwise）验证：保存 npy + md5sum | `reference.md` §17.1；`checklists.md` 7 |
| `ACLNN算子适配指导/基于 AI 工具Cursor进行pytorch算子分析.md` | 用提示词定位 PyTorch 正反向实现路径 | `reference.md` 18；`examples.md` 对标分析场景 |
| *(用户直接指示 + op-plugin 源码)* | PTA 源码审查方法：三件套文件、差异识别、不一致时的处理流程 | `reference.md` §25；`SKILL.md` 信息收集清单；`checklists.md` §0a |

## 备注
- 本映射强调"高频可复用"信息；单个文档中的示例代码/截图不做逐条搬运，避免 skill 过长。
- **vmap 相关内容**：源文档 `4. 算子关键特性.md` 中有 vmap 注册与测试要点。当前 skill 将 vmap 作为
  **可选扩展**放在 `reference.md` §23，仅在目标算子需要 vmap 支持时启用。如果后续 vmap 成为必选交付项，
  需要在 `checklists.md` 中增加专项检查清单。
- **§25 PTA 源码审查**：该章节来源不是 18 篇原始文档，而是用户在 skill 审查过程中直接指示添加的，
  基于 op-plugin 仓库的实际代码（`derivatives.yaml`、`op_plugin_functions.yaml`、C++ 实现）。
