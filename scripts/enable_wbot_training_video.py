"""Switch both running jobs to video recording at their next saved checkpoint."""
from pathlib import Path
import os, signal, subprocess, time, json
ROOT=Path('/home/songhat/code_workspace/unitree_rl_mjlab')
RUN=ROOT/'logs/launch_20261001'
PYTHON='/home/songhat/miniconda3/envs/env_mjlab/bin/python'
TASKS={'Wbot-12Dof-Flat':'wbot_12dof_velocity','Wbot-12Dof-Suspended':'wbot_12dof_suspended'}
jobs={}
for task,exp in TASKS.items():
 old=max((ROOT/'logs/rsl_rl'/exp).glob('*_delay20ms_energy'),key=lambda p:p.stat().st_mtime)
 checkpoints=list(old.glob('model_*.pt'))
 latest=max(int(p.stem.split('_')[-1]) for p in checkpoints)
 jobs[task]=(old,latest,int((RUN/f'{task}.pid').read_text()))
while jobs:
 for task,(old,latest,pid) in list(jobs.items()):
  candidates=[p for p in old.glob('model_*.pt') if int(p.stem.split('_')[-1])>latest]
  if not candidates:continue
  checkpoint=max(candidates,key=lambda p:int(p.stem.split('_')[-1]))
  # Wait for checkpoint and corresponding ONNX export to finish.
  if time.time()-checkpoint.stat().st_mtime < 8:continue
  os.kill(pid,signal.SIGINT)
  for _ in range(100):
   if not Path(f'/proc/{pid}').exists():break
   time.sleep(.1)
  else:os.kill(pid,signal.SIGTERM);time.sleep(1)
  command=[PYTHON,'-u','scripts/train.py',task,'--env.scene.num-envs','2048','--agent.max-iterations','10001','--agent.logger','tensorboard','--agent.run-name','delay20ms_energy_video','--checkpoint-file',str(checkpoint),'--video','True','--video-interval','2000','--video-length','200','--video-width','640','--video-height','360']
  with open(RUN/f'{task}.log','ab',buffering=0) as log:
   process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'})
  (RUN/f'{task}.pid').write_text(str(process.pid)+'\n')
  print(task,'resumed',checkpoint,'pid',process.pid,flush=True)
  del jobs[task]
 time.sleep(2)
print('Both tasks now record 200-frame videos every 2000 policy steps.',flush=True)
