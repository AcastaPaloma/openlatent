#!/usr/bin/env python3
"""B' kill test: self-rendered MetaWorld scripted episodes with exact ground truth (CPU only, OSMesa).

Per episode we store: clean frames (T+1,H,W,3), exact arm / object silhouettes from MuJoCo segmentation
(packed bits), macro actions (mean of K executed micro-actions, the label a latent action should decode),
EE pixel position, EE/object 3-D position. Distractors are NOT baked in: they are deterministic functions of
(actions, seed), see distractors.py, so every distractor type shares the same clean frames and the
"distractor-only layer" is exact.

Usage: python envgen.py --out ~/work/data/b_mw --train 600 --val 100 --test 100 --procs 48
"""
import os, sys, time, json, argparse, warnings
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
import numpy as np

TASKS = ["reach-v3", "push-v3", "pick-place-v3"]
H = W = 128
K = 4          # micro env steps per macro step
T = 16         # macro steps per episode (64 env steps)
CAM = dict(azimuth=150.0, elevation=-50.0, distance=1.1, lookat=(0.0, 0.68, 0.05))  # free camera, workspace fills the view
ACT_NOISE = 0.5   # std of per-macro-step exploration noise on xyz (so actions are not a function of state)
SPLITS = {"train": 0, "val": 10000, "test": 20000}

_ENV = {}
_ENV_REND = None


def _policy(task):
    from metaworld.policies import SawyerReachV3Policy, SawyerPushV3Policy, SawyerPickPlaceV3Policy
    return {"reach-v3": SawyerReachV3Policy, "push-v3": SawyerPushV3Policy, "pick-place-v3": SawyerPickPlaceV3Policy}[task]()


def _get(task):
    if task not in _ENV:
        import gymnasium as gym, metaworld, mujoco  # noqa
        env = gym.make("Meta-World/MT1", env_name=task)
        u = env.unwrapped
        m = u.model
        arm = np.zeros(m.nbody, bool)
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_arm_base_link")
        for b in range(m.nbody):
            p = b
            while p != 0:
                if p == root:
                    arm[b] = True
                    break
                p = m.body_parentid[p]
        objb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
        geom_arm = arm[m.geom_bodyid]
        geom_obj = m.geom_bodyid == objb
        rend = mujoco.Renderer(m, H, W)
        rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0       # speed (software GL); shadows are not needed
        rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = CAM["azimuth"], CAM["elevation"], CAM["distance"]
        cam.lookat[:] = CAM["lookat"]
        _ENV[task] = (env, u, rend, geom_arm, geom_obj, cam, _policy(task))
    return _ENV[task]


def project(u, cam, p3):
    """world -> pixel for the free camera (pinhole); pose is read from the scene after update_scene."""
    gl = _ENV_REND.scene.camera[0]
    pos, fwd, up = np.array(gl.pos), np.array(gl.forward), np.array(gl.up)
    right = np.cross(fwd, up)
    q = np.asarray(p3) - pos
    f = 0.5 * H / np.tan(np.deg2rad(u.model.vis.global_.fovy) / 2)
    return np.array([W / 2 + f * (q @ right) / (q @ fwd), H / 2 - f * (q @ up) / (q @ fwd)])


def render(task):
    global _ENV_REND
    env, u, rend, geom_arm, geom_obj, cam, _ = _get(task)
    _ENV_REND = rend
    rend.disable_segmentation_rendering()
    rend.update_scene(u.data, camera=cam)
    img = rend.render().copy()
    rend.enable_segmentation_rendering()
    rend.update_scene(u.data, camera=cam)
    seg = rend.render()
    rend.disable_segmentation_rendering()
    gid = seg[..., 0]
    isgeom = seg[..., 1] == 5  # mjOBJ_GEOM
    safe = np.where(isgeom & (gid >= 0), gid, 0)
    arm = isgeom & geom_arm[safe]
    obj = isgeom & geom_obj[safe]
    return img, arm, obj


def gen_episode(args):
    task, seed = args
    env, u, rend, geom_arm, geom_obj, cam, pol = _get(task)
    rng = np.random.default_rng(seed + 977)
    obs, _ = env.reset(seed=int(seed))
    imgs, arms, objs, ee3, obj3, eepix = [], [], [], [], [], []

    def snap():
        i, a, o = render(task)
        imgs.append(i); arms.append(a); objs.append(o)
        e = u.tcp_center.copy()
        ee3.append(e); obj3.append(obs[4:7].copy()); eepix.append(project(u, cam, e))

    snap()
    acts, micro = [], []
    for t in range(T):
        n = rng.normal(0, ACT_NOISE, 3)
        mk = []
        for k in range(K):
            a = np.clip(pol.get_action(obs) + np.r_[n, 0.0], -1, 1).astype(np.float32)
            obs, _, _, _, _ = u.step(a)
            mk.append(a)
        mk = np.stack(mk)
        micro.append(mk)
        acts.append(mk.mean(0))
        snap()
    return dict(task=task, seed=seed, frames=np.stack(imgs).astype(np.uint8),
                arm=np.packbits(np.stack(arms), axis=-1), obj=np.packbits(np.stack(objs), axis=-1),
                acts=np.stack(acts).astype(np.float32), micro=np.stack(micro).astype(np.float32),
                ee3=np.stack(ee3).astype(np.float32), obj3=np.stack(obj3).astype(np.float32),
                eepix=np.stack(eepix).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/work/data/b_mw"))
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = a.tasks.split(",")
    import multiprocessing as mp
    t0 = time.time()
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        jobs = [(tk, SPLITS[split] + i) for tk in tasks for i in range(n)]
        with mp.get_context("fork").Pool(a.procs) as pool:
            eps = pool.map(gen_episode, jobs, chunksize=2)
        keys = ["frames", "arm", "obj", "acts", "micro", "ee3", "obj3", "eepix"]
        d = {k: np.stack([e[k] for e in eps]) for k in keys}
        d["task"] = np.array([tasks.index(e["task"]) for e in eps], np.int16)
        d["seed"] = np.array([e["seed"] for e in eps], np.int64)
        np.savez(os.path.join(a.out, f"{split}.npz"), **d)
        print(split, {k: v.shape for k, v in d.items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(tasks=tasks, H=H, W=W, K=K, T=T, cam=CAM, act_noise=ACT_NOISE, splits=SPLITS),
              open(os.path.join(a.out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
