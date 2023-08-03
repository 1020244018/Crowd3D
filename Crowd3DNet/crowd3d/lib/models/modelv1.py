from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from models.base import Base
from models.CoordConv import get_coord_maps
from models.basic_modules import BasicBlock,Bottleneck

import config
from config import args
from loss_funcs import Loss
from maps_utils.result_parser import ResultParser
from utils import BHWC_to_BCHW

BN_MOMENTUM = 0.1

class CROWD3DNET(Base):
    def __init__(self, backbone=None,**kwargs):
        super(CROWD3DNET, self).__init__()
        self.backbone = backbone
        self._result_parser = ResultParser()
        self._build_head()
        self._build_cam_layers()
        if args().model_return_loss:
            self._calc_loss = Loss()
        if not args().fine_tune and not args().eval:
            self.init_weights()
            self.backbone.load_pretrain_params()

    def head_forward(self,x, ground_cam_variable):
        #generate heatmap (K) + associate embedding (K)
        #kp_heatmap_ae = self.final_layers[0](x)

        x = torch.cat((x, self.coordmaps.to(x.device).repeat(x.shape[0],1,1,1)), 1)
        ground_cam_variable_x=ground_cam_variable.unsqueeze(-1).unsqueeze(-1)
        ground_cam_variable_x=ground_cam_variable_x.tile(1, 1, 128, 128)
        x_with_gn=torch.cat((x, ground_cam_variable_x), 1)
        params_maps = self.final_layers[1](x_with_gn)

        # 前3列设为delta3d
        delta3d_maps=params_maps[:, :3]
        params_maps[:, :3]=torch.sigmoid(delta3d_maps)

        center_maps = self.final_layers[2](x)
        hvip2d_maps = self.final_layers[3](x)
        tc_offset_maps = self.final_layers[4](x)
    
        params_maps = torch.cat([hvip2d_maps, tc_offset_maps, params_maps], 1)
        output = {'params_maps':params_maps.float(), 'center_map':center_maps.float()} #, 'kp_ae_maps':kp_heatmap_ae.float()
        return output

    def _build_head(self):
        self.outmap_size = args().centermap_size
        params_num = self._result_parser.params_map_parser.params_num
        self.head_cfg = {'NUM_HEADS': 1, 'NUM_CHANNELS': 64, 'NUM_BASIC_BLOCKS': args().head_block_num}
        self.output_cfg = {'NUM_PARAMS_MAP':params_num + 3, 'NUM_CENTER_MAP':1, 'NUM_HVIP2D_MAP':args().hvip_dim, 'NUM_OFFSET': 2} 
        self.final_layers = self._make_final_layers(self.backbone.backbone_channels)
        self.coordmaps = get_coord_maps(128)
        self.downsample = nn.Upsample(size=(128,128), mode='nearest', align_corners=None)
    
    def _build_cam_layers(self):
        cams_layers_list=[]
        cams_layers_list.append(nn.Linear(3, 8))
        cams_layers_list.append(nn.ReLU(inplace=True))
        cams_layers_list.append(nn.Linear(8, 16))
        cams_layers_list.append(nn.ReLU(inplace=True))
        cams_layers_list.append(nn.Linear(16, 8))
        self.cams_layers=nn.Sequential(*cams_layers_list)

        ground_layers_list=[]
        ground_layers_list.append(nn.Linear(4, 8))
        ground_layers_list.append(nn.ReLU(inplace=True))
        ground_layers_list.append(nn.Linear(8, 16))
        ground_layers_list.append(nn.ReLU(inplace=True))
        ground_layers_list.append(nn.Linear(16, 8))
        self.ground_layers=nn.Sequential(*ground_layers_list)

    def _make_final_layers(self, input_channels):
        final_layers = []
        final_layers.append(None)
        #output_channels = self.NUM_JOINTS + self.NUM_JOINTS
        #final_layers.append(nn.Conv2d(in_channels=input_channels,out_channels=output_channels,\
        #    kernel_size=1,stride=1,padding=0))

        input_channels += 2
        if args().merge_smpl_camera_head:
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_PARAMS_MAP']+self.output_cfg['NUM_HVIP2D_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
        else:
            final_layers.append(self._make_head_layers(input_channels+16, self.output_cfg['NUM_PARAMS_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_CENTER_MAP']))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_HVIP2D_MAP'], end_sigmoid=True))
            final_layers.append(self._make_head_layers(input_channels, self.output_cfg['NUM_OFFSET'], end_sigmoid=True)) # for tc offset.

        return nn.ModuleList(final_layers)
    
    def _make_head_layers(self, input_channels, output_channels, end_sigmoid=False):
        head_layers = []
        num_channels = self.head_cfg['NUM_CHANNELS']

        kernel_sizes, strides, paddings = self._get_trans_cfg()
        for kernel_size, padding, stride in zip(kernel_sizes, paddings, strides):
            head_layers.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                    nn.BatchNorm2d(num_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)))
        
        for i in range(self.head_cfg['NUM_HEADS']):
            layers = []
            for _ in range(self.head_cfg['NUM_BASIC_BLOCKS']):
                layers.append(nn.Sequential(BasicBlock(num_channels, num_channels)))
            head_layers.append(nn.Sequential(*layers))

        head_layers.append(nn.Conv2d(in_channels=num_channels,out_channels=output_channels,\
            kernel_size=1,stride=1,padding=0))
        if end_sigmoid:
            head_layers.append(nn.Sigmoid())
        return nn.Sequential(*head_layers)

    def _get_trans_cfg(self):
        if self.outmap_size == 32:
            kernel_sizes = [3,3]
            paddings = [1,1]
            strides = [2,2]
        elif self.outmap_size == 64:
            kernel_sizes = [3]
            paddings = [1]
            strides = [2]
        elif self.outmap_size == 128:
            kernel_sizes = [3]
            paddings = [1]
            strides = [1]

        return kernel_sizes, strides, paddings
    
    def _make_gn_layer(self, input_channels, output_channels): 
        layers=[]
        layer_1=nn.Sequential( # 512 * 512 -> 128, 128
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=4,
                padding=1),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True))
        layers.append(layer_1)
        layers.append(nn.Sequential(BasicBlock(output_channels, output_channels))) # residual
        return nn.Sequential(*layers)
    
    def _make_mix_layer(self, input_channels, output_channels):
        layers=[]
        layer_1=nn.Sequential( # 4 * 512 * 512 -> 2 * 512 * 512
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True))
        layers.append(layer_1)

        return nn.Sequential(*layers)
    
    def feed_forward(self, meta_data):
        x = self.backbone(meta_data['image'].contiguous().cuda()) # 30, 32, 128, 128
        
        # ground_cam_params
        ground, camK, patch_leftTop, offsets=meta_data['ground'], meta_data['camK'], meta_data['patch_leftTop'], meta_data['offsets']
        img_pad_size, crop_trbl, pad_trbl, data_scale, scene_shape = offsets[:, : 2], offsets[:,2:6], offsets[:, 6:10], offsets[:, 10], offsets[:, 11:12]
        data_scale=data_scale.unsqueeze(-1)
        leftTop = torch.stack(
        [crop_trbl[:, 3] - pad_trbl[:, 3], crop_trbl[:, 0] - pad_trbl[:, 0]],
        1)
        orign_size=img_pad_size * data_scale
        leftTop=leftTop * data_scale
        fx, fy, cx, cy = camK[:, 0, 0], camK[:, 1, 1], camK[:, 0, 2], camK[:, 1, 2]
        cx_cy=torch.stack([cx, cy], 1)
        transed_cx_cy=cx_cy - leftTop - patch_leftTop
        transed_cx_cy=transed_cx_cy / orign_size
        fov=(fx / scene_shape[0]).unsqueeze(-1)

        cam_para_input=torch.cat([fov, transed_cx_cy], 1).float()
        cam_variable=self.cams_layers(cam_para_input.contiguous().cuda().float())
        
        N=ground[:, :3]
        mo=torch.norm(N, dim=1)
        ground_norm=ground / mo.unsqueeze(-1)

        ground_variable=self.ground_layers(ground_norm.contiguous().cuda().float())
        ground_cam_variable=torch.cat([cam_variable, ground_variable], 1)
        outputs = self.head_forward(x, ground_cam_variable)
        return outputs
    

if __name__ == '__main__':
    args().configs_yml = 'configs/v1.yml'
    args().model_version=1
    from models.build import build_model
    model = build_model().cuda()
    outputs=model.feed_forward({'image':torch.rand(4,512,512,3).cuda()})
    for key, value in outputs.items():
        if isinstance(value,tuple):
            print(key, value)
        elif isinstance(value,list):
            print(key, value)
        else:
            print(key, value.shape)