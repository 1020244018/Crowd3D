import os
project_root=os.path.dirname(os.path.abspath(__file__))

# set the abs alphapose folder
alphapose_root='/data/wh/workspace/competition_code/tools/AlphaPose-pytorch-1.11/'

estimate_ground_root=os.path.join(project_root, 'estimate_ground_faster')

crowd3dnet_root=os.path.join(project_root, 'Crowd3DNet')


smpl_model_root=os.path.join(crowd3dnet_root, 'public_params', 'model_data', 'parameters', 'smpl')