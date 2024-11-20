import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import  uv_to_grid
from model.modules  import UnetNoCond5DS, GeomConvLayers, GeomConvBottleneckLayers, ShapeDecoder, ColorDecoder, ScaleDecoder


class POP_no_unet(nn.Module):
    def __init__(
                self, 
                c_geom=64,
                geom_layer_type='conv',
                nf=64,
                hsize=256,
                up_mode='upconv',
                use_dropout=False,
                uv_feat_dim=2,
                ):

        super().__init__()
        self.geom_layer_type = geom_layer_type 
        
        geom_proc_layers = {
            'unet': UnetNoCond5DS(c_geom, c_geom, nf, up_mode, use_dropout),
            'conv': GeomConvLayers(c_geom, c_geom, c_geom, use_relu=False),
            'bottleneck': GeomConvBottleneckLayers(c_geom, c_geom, c_geom, use_relu=False),
        }


        if geom_layer_type is not None:
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]
    

        self.decoder = ShapeDecoder(in_size=uv_feat_dim + c_geom,
                                    hsize=hsize, actv_fn='softplus')
                                    
        self.colordecoder = ColorDecoder(hsize=hsize, actv_fn='softplus')
        self.scaledecoder = ScaleDecoder(hsize=hsize, actv_fn='softplus')

            
    def forward(self, pose_featmap, geom_featmap, uv_loc, image, normal):
        '''
        :param x: input posmap, [batch, 3, 256, 256]
        :param geom_featmap: a [B, C, H, W] tensor, spatially pixel-aligned with the pose features extracted by the UNet
        :param uv_loc: querying uv coordinates, ranging between 0 and 1, of shape [B, H*W, 2].
        :param pq_coords: the 'sub-UV-pixel' (p,q) coordinates, range [0,1), shape [B, H*W, 1, 2]. 
                        Note: It is the intra-patch coordinates in SCALE. Kept here for the backward compatibility with SCALE.
        :return:
            clothing offset vectors (residuals) and normals of the points
        '''

        if self.geom_layer_type is not None:
                geom_featmap = self.geom_proc_layers(geom_featmap)
        
        if  pose_featmap is None:


            pix_feature = geom_featmap
        else:
            pix_feature = pose_featmap + geom_featmap

        
        feat_res = geom_featmap.shape[2]
        uv_res = int(uv_loc.shape[1]**0.5)


        if feat_res != uv_res:
            query_grid = uv_to_grid(uv_loc, uv_res)
            pix_feature = F.grid_sample(pix_feature, query_grid, mode='bilinear', align_corners=False)


        B, C, H, W = pix_feature.shape
        N_subsample = 1

        uv_feat_dim = uv_loc.size()[-1]
        uv_loc = uv_loc.expand(N_subsample, -1, -1, -1).permute([1, 2, 0, 3])


        pix_feature = pix_feature.view(B, C, -1).expand(N_subsample, -1,-1,-1).permute([1,2,3,0])
        pix_feature = pix_feature.reshape(B, C, -1)

        uv_loc = uv_loc.reshape(B, -1, uv_feat_dim).transpose(1, 2)


        residuals, _, _ = self.decoder(torch.cat([pix_feature, uv_loc], 1))
        shs = self.colordecoder(image, torch.cat([pix_feature, uv_loc], dim=1), normal)
        scales = self.scaledecoder(normal, torch.cat([pix_feature, uv_loc], dim=1))
        
        return residuals, scales, shs 

