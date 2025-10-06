#!/usr/bin/env python3
# convert_bin_to_safetensors.py
import sys
import torch
from safetensors.torch import save_file

def load_state_dict(bin_path):
    obj = torch.load(bin_path, map_location="cpu")
    # 常见情形：obj 就是 state_dict（mapping str->Tensor）
    if isinstance(obj, dict):
        # 直接是 tensor 映射
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        # 常见的包装字段
        for key in ("state_dict", "model", "module"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        # 尝试在子项里找到第一个满足条件的 dict
        for k, v in obj.items():
            if isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()):
                print(f"Using nested dict '{k}' as state_dict")
                return v
    raise ValueError("无法在文件中定位到 state_dict（mapping str->Tensor）。")

def normalize_state_dict(sd: dict):
    new = {}
    for k, v in sd.items():
        if isinstance(v, torch.nn.Parameter):
            v = v.data
        if not isinstance(v, torch.Tensor):
            try:
                v = torch.tensor(v)
            except Exception as e:
                raise TypeError(f"key={k} 的值不是 Tensor，且无法转换：{type(v)}") from e
        new[k] = v.cpu()
    return new

def main(bin_path, dst_path):
    print("加载：", bin_path)
    sd = load_state_dict(bin_path)
    print("keys 数量：", len(sd))
    sd_cpu = normalize_state_dict(sd)
    print("保存为 safetensors：", dst_path)
    save_file(sd_cpu, dst_path)
    print("完成。")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python convert_bin_to_safetensors.py pytorch_model.bin model.safetensors")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
