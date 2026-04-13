# Memory Consistency Issue Cases

Historical NPU vs GPU memory consistency issues and their root cause analysis.

Format:
```yaml
- issue_number: "[ISSUE defect number]"
  description: "[brief issue description]"
  aclnn_interface: "[related aclnn interface or note]"
  root_cause: "[root cause analysis]"
  solution: "[resolution strategy]"
  category: "[issue classification]"
```

## Cases

### INT-20260102-630 - aclnnInplaceNormal
- issue_number: "INT-YYYYMMDD-NNN"
- description: "torch.randn memory consumption is 3x that of torch_gpu due to internal cast in aclnnInplaceNormal"
- aclnn_interface: "aclnnInplaceNormal"
- root_cause: "aclnn interface inserts an internal cast operation"
- solution: "File requirement to add a non-cast code path"
- category: "internal-cast-overhead"

### INT-20260202-231 - missing aclnn kernel
- issue_number: "INT-20260202-231"
- description: "torch.linalg.solve memory consistency exceeds 5% gap vs GPU due to missing dedicated NPU backend"
- aclnn_interface: "missing dedicated aclnn kernel"
- root_cause: "aten::_linalg_solve_ex.result and aten::linalg_solve_backward lack dedicated NPU backend implementations"
- solution: "File requirement to develop dedicated aclnn kernels"
- category: "missing-aclnn-kernel"