s = open("train_arms.py").read()
old = "    return enc.eval()\n"
assert s.count(old) == 1; s = s.replace(old, "    global TASK_HEAD; TASK_HEAD = task\n    return enc.eval()\n", 1)
old = "    with torch.no_grad(): Z = enc(Xt).cpu().numpy()\n"
assert s.count(old) == 1
new = old + '''    if os.environ.get("SAVEW"):  # export the trained adapter (safetensors, no pickle) with the input standardisation it needs
        from safetensors.torch import save_file
        tens = {"encoder." + k: v.detach().cpu().contiguous() for k, v in enc.state_dict().items()}
        tens.update({"task_head." + k: v.detach().cpu().contiguous() for k, v in TASK_HEAD.state_dict().items()})
        tens["input_mean"] = torch.tensor(mu, dtype=torch.float32); tens["input_std"] = torch.tensor(sd, dtype=torch.float32)
        os.makedirs(f"{D}/weights", exist_ok=True); save_file(tens, f"{D}/weights/{ARM}_s{SEED}_{STEPS}steps.safetensors"); print("saved weights", ARM, SEED, flush=True)
'''
s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
