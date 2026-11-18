"""Derive a physical 12-DoF model by locking arms in the reference 20-DoF XML."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'src/assets/robots/wbot/xmls/wbot_20dof.xml'
DEST=ROOT/'src/assets/robots/wbot/xmls/wbot_12dof.xml'

def build():
 tree=ET.parse(REFERENCE)
 angles={'shoulder_pitch':-.087,'shoulder_roll':.297,'shoulder_yaw':-.442,'elbow':1.097}
 for side in ('left','right'):
  for part,angle in angles.items():
   body=tree.find(f'.//body[@name="{side}_{part}_link"]')
   joint=body.find('joint')
   if side=='right' and part in ('shoulder_roll','shoulder_yaw'):angle=-angle
   quat=np.fromstring(body.get('quat','1 0 0 0'),sep=' ')
   axis=np.fromstring(joint.get('axis'),sep=' ')
   rotation=Rotation.from_quat(np.roll(quat,-1))*Rotation.from_rotvec(axis*angle)
   body.set('quat',' '.join(f'{v:.12g}' for v in np.roll(rotation.as_quat(),1)))
   body.remove(joint)
 ET.indent(tree,space='    ');tree.write(DEST,encoding='unicode')
 DEST.write_text(DEST.read_text()+'\n')
if __name__=='__main__':build()
