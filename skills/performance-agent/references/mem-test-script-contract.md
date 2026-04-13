# Memory Test Script Contract

Rules for agent-generated NPU / GPU memory test scripts.

## Requirements

1. Set up the device, create input tensors with user-provided shapes,
   call the target API
2. Measure memory using the platform-specific APIs below
3. Print **exactly one JSON line** to stdout

## JSON output keys

| Key | Type | Description |
|-----|------|-------------|
| `target_api` | string | Full API path, e.g. `"torch.linalg.solve"` |
| `32aligned` | bool | `true` when **every** dimension of **every** input tensor is a multiple of 32, `false` otherwise |
| `total_driver_GB` | float | Driver-level memory delta (before − after) |
| `pta_reserved_GB` / `gpu_reserved_GB` | float | CachingAllocator reserved memory |
| `pta_activated_GB` / `gpu_activated_GB` | float | CachingAllocator peak allocated memory |

Use `pta_*` keys for NPU scripts, `gpu_*` keys for GPU scripts.

## NPU measurement

```python
device_id = torch.npu.current_device()
torch_npu.npu.reset_peak_memory_stats()
torch_npu.npu.reset_max_memory_allocated()
mem_before = torch.npu.mem_get_info(device_id)
output = <api_call>
mem_after = torch.npu.mem_get_info(device_id)
total_driver_GB  = (mem_before[0] - mem_after[0]) / 1024**3
pta_reserved_GB  = torch.npu.memory_reserved(device_id) / 1024**3
pta_activated_GB = torch.npu.max_memory_allocated(device_id) / 1024**3
print(json.dumps({
    "target_api": "<api>", "32aligned": <bool>,
    "total_driver_GB": total_driver_GB,
    "pta_reserved_GB": pta_reserved_GB,
    "pta_activated_GB": pta_activated_GB,
}))
```

## GPU measurement

```python
device_id = torch.cuda.current_device()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_max_memory_allocated()
mem_before = torch.cuda.mem_get_info(device_id)
output = <api_call>
mem_after = torch.cuda.mem_get_info(device_id)
total_driver_GB  = (mem_before[0] - mem_after[0]) / 1024**3
gpu_reserved_GB  = torch.cuda.memory_reserved(device_id) / 1024**3
gpu_activated_GB = torch.cuda.max_memory_allocated(device_id) / 1024**3
print(json.dumps({
    "target_api": "<api>", "32aligned": <bool>,
    "total_driver_GB": total_driver_GB,
    "gpu_reserved_GB": gpu_reserved_GB,
    "gpu_activated_GB": gpu_activated_GB,
}))
```

## Distributed APIs

Initialise the process group (`hccl` for NPU, `nccl` for GPU) before
device setup.
