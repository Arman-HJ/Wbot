#!/usr/bin/env python3
"""Loop a recorded actuator-command rollout in the native MuJoCo viewer.

Each episode assigns its initial state once. All subsequent motion comes from
mj_step with the recorded, torque-limited joint commands. Space pauses; R restarts.
"""
import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--physics', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--hold-final', action='store_true', help='Keep simulating the final joint target until R or window close; do not automatically restart')
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error('speed must be positive')
    motion = np.load(args.motion, allow_pickle=False)
    physics = np.load(args.physics, allow_pickle=False)
    loader = mujoco.MjModel.from_binary_path if args.model.suffix == '.mjb' else mujoco.MjModel.from_xml_path
    model = loader(str(args.model.resolve()))
    data = mujoco.MjData(model)
    dt = float(np.asarray(physics['timestep']).ravel()[0])
    if not np.isclose(dt, model.opt.timestep, rtol=0, atol=1e-12):
        raise ValueError('Stored controls and model have different timesteps')
    stride = round(1 / (float(motion['fps'].ravel()[0]) * dt))
    ui = dict(paused=False, restart=False)

    def key(code):
        if code == 32:
            ui['paused'] = not ui['paused']
        elif code in (82, 114):
            ui['restart'] = True

    print('Native forward dynamics: Space = pause/resume, R = restart, close window = exit.', flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key) as viewer:
        viewer.cam.distance = 1.65
        viewer.cam.elevation = -12
        viewer.cam.azimuth = 100
        viewer.cam.lookat[:] = [.15, 0, .3]
        while viewer.is_running():
            mujoco.mj_resetData(model, data)
            data.qpos[:] = motion['qpos'][0]
            data.qvel[:] = motion['qvel'][0]
            mujoco.mj_forward(model, data)
            ui['restart'] = False
            error = 0.
            viewer.sync()
            start = time.monotonic()
            while viewer.is_running() and time.monotonic() - start < 1.:
                viewer.sync()
                time.sleep(.02)
            for i, command in enumerate(physics['ctrl']):
                if not viewer.is_running() or ui['restart']:
                    break
                while ui['paused'] and viewer.is_running() and not ui['restart']:
                    viewer.sync()
                    time.sleep(.02)
                if not viewer.is_running() or ui['restart']:
                    break
                tick = time.monotonic()
                if i % stride == 0:
                    mujoco.mj_forward(model, data)
                data.ctrl[:] = command
                data.qfrc_applied[:] = 0
                data.xfrc_applied[:] = 0
                mujoco.mj_step(model, data)
                error = max(error, float(np.max(np.abs(data.qpos - physics['qpos'][i]))))
                if (i + 1) % stride == 0:
                    viewer.sync()
                time.sleep(max(0, dt / args.speed - (time.monotonic() - tick)))
            print(f'Episode replay max qpos error: {error:.3g}', flush=True)
            if args.hold_final:
                while viewer.is_running() and not ui['restart']:
                    tick = time.monotonic()
                    if not ui['paused']:
                        for _ in range(stride):
                            data.ctrl[:] = physics['ctrl'][-1]
                            data.qfrc_applied[:] = 0
                            data.xfrc_applied[:] = 0
                            mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0, stride * dt / args.speed - (time.monotonic() - tick)))
                continue
            start = time.monotonic()
            while viewer.is_running() and not ui['restart'] and time.monotonic() - start < 2.:
                viewer.sync()
                time.sleep(.02)


if __name__ == '__main__':
    main()
