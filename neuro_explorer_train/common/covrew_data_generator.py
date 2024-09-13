
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import random
sys.path.insert(0, '/home/hankm/python_ws/neuro_ae/utils')
from data_augmentation import rotate, flip
from numpy import loadtxt
from dataclasses import dataclass
import math
import fnmatch

@dataclass
class MetaData:
    # rx  ry  qx  qy  qz  qw  tx  ty  gm_height   gm_width    gm_ox   gm_oy   resolution  OFFSET  roi.x   roi.y   roi.height  roi.width
    gm_rx: float = None
    gm_ry: float = None
    qx: float = 0.
    qy: float = 0.
    qz: float = 0.
    qw: float = 0.
    gm_tx: float = 0.
    gm_ty: float = 0.
    gmheight: float = None
    gmwidth: float = None
    ox: float = None
    oy: float = None
    res: float = None
    offset: int = None
    roi_x: int = None
    roi_y: int = None
    roi_h: int = None
    roi_w: int = None

class data_generator():
    def __init__(self, name, **kwargs):
        self.name = name
        self.dataset_dir = kwargs['data_configs']['dataset_dir']
        self.gmheight           = int( kwargs['data_configs']['gmheight'] )
        self.gmwidth            = int( kwargs['data_configs']['gmwidth'] )
        self.gmheight_p         = int( kwargs['data_configs']['gmheight_p'] )
        self.gmwidth_p          = int( kwargs['data_configs']['gmwidth_p'] )
        assert(self.gmheight == self.gmwidth)
        self.pad_size           = int( (self.gmwidth_p - self.gmwidth) / 2 ) #  int( kwargs['data_configs']['pad_size'] )
        self.num_rounds         = int( kwargs['data_configs']['num_rounds'])
        self.metalen            = kwargs['data_configs']['metalen']
        self.worlds             = kwargs['data_configs']['worlds']
        # network configs
        self.output_channels    = int(kwargs['network_configs']['output_channels'])
        self.input_channels     = int(kwargs['network_configs']['input_channels'])
        self.batch_size         = int(kwargs['network_configs']['batch_size'])

    def pad_map(self, input_map, pad_size, base_val = 0):
        (H, W, C) = input_map.shape # input_map is numpy
        if base_val == 0:
            img_pad = np.zeros([H + pad_size * 2, W + pad_size * 2, C], dtype=np.float32)
        else:
            img_pad = np.ones([H + pad_size * 2, W + pad_size * 2, C], dtype=np.float32) * float(base_val)
        img_pad[pad_size:-pad_size, pad_size:-pad_size, :] = input_map
        return img_pad
    def add_gaussian_to_pixel(self, input_img, xc, yc, sigma = 1., kernel_size = 5):
        assert( kernel_size % 2 == 1 )
        rows, cols = input_img.shape
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
        dst = np.sqrt(x**2 + y**2)
        normal = 1/math.sqrt(2.0 * np.pi * sigma**2)
        gauss = np.exp(-(dst - 0) **2 / (2.0 * sigma**2)) * normal
        #normalize
        cidx = int((gauss.shape[0]-1 )/2)
        gauss = gauss / np.max(gauss) #gauss[cidx, cidx]
        input_img_rep0 = np.concatenate( (input_img, input_img, input_img), axis=0 )
        input_img_rep = np.concatenate( (input_img_rep0, input_img_rep0, input_img_rep0), axis=1)
        kwidth = cidx
        ys = int(yc) - kwidth + rows
        ye = int(yc) + kwidth + rows
        xs = int(xc) - kwidth + cols
        xe = int(xc) + kwidth + cols
        input_img_rep[ys:ye+1, xs:xe+1] = gauss
        out_img = input_img_rep[rows:rows*2, cols:cols*2] # cropping back to the org size
        out_img = out_img[..., np.newaxis]
        return out_img
    def load_padded_map(self, mapfile, pad_size, base_val=0):
        '''
        :param mapfile:
        :return: a padded HxWxC img
        '''
        img_raw = cv2.imread(mapfile, -1).astype(np.float32)
        if img_raw.ndim == 2:
            img_raw = img_raw[:, :, np.newaxis]
        img_pad = self.pad_map(img_raw, pad_size, base_val)
        return img_pad

    def load_padded_txt(self, potmapfile, pad_size, base_val, inv_map=False):
        '''
        :param mapfile:
        :return: a padded HxWxC img
        '''
        img_raw = loadtxt(potmapfile, delimiter=",")
        # normalize img_raw
        img_max = np.max(img_raw)
        if img_max == 0:
            img_norm = img_raw  # all zeros
        else:
            img_norm = img_raw / np.max(img_raw)
        if img_norm.ndim == 2:
            img_norm = img_norm[:, :, np.newaxis]
        img_pad = self.pad_map(img_norm, pad_size, base_val)
        # img_pad = np.where(img_pad < 0, 1, img_pad)  # lets re-map (failed path plans and NON FR cells) to max cost
        if inv_map:  # invert map
            img_pad = np.where(img_pad <= 0, 1., img_pad)
            img_pad = 1. - img_pad
        else:
            img_pad = np.where(img_pad < 0, 0, img_pad)
        max_img_pad = np.max(img_pad)
        img_out = (
            img_pad / max_img_pad if max_img_pad > 0 else img_pad)  # to make sure the output img is distributed on 0 ~ 1.
        return img_out
    def load_metadata(self, metadata_raw_file ):
        # rx  ry  qx  qy  qz  qw  tx  ty  gm_height   gm_width    gm_ox   gm_oy   resolution  OFFSET  roi.x   roi.y   roi.height  roi.width
        (rx_w, ry_w, qx, qy, qz, qw, tx_w, ty_w, height, width, ox, oy, res, offset, roi_x, roi_y, roi_h, roi_w) = loadtxt( metadata_raw_file )
        metadata = MetaData()
        metadata.gm_rx = int( ( rx_w - ox ) / res )
        metadata.gm_ry = int( ( ry_w - oy ) / res )
        metadata.gm_tx = int( ( tx_w - ox ) / res )
        metadata.gm_ty = int( ( ty_w - oy ) / res )
        metadata.gmheight = height
        metadata.gmwidth  = width
        metadata.roi_x = roi_x
        metadata.roi_y = roi_y
        metadata.roi_h = roi_h
        metadata.roi_w = roi_w
        metadata.offset = offset
        metadata.ox = ox
        metadata.oy = oy
        metadata.res = res
        return metadata
    def transform_map_to_robotposition( self, input_map, rx, ry, base_val ):
        '''
        :param input_img:   0~255 intensity
        :param rx, ry:       robot position in input_img
        :return:            transformed (robot centered) input img
        '''
        assert(input_map.ndim == 3)
        (H, W, C) = input_map.shape
        half_size = H / 2
    #    ry = pt_yx[0][0]
    #    rx = pt_yx[1][0]
        map_expanded = np.ones([H*3, W*3, C], dtype=np.float32) * base_val
        map_expanded[H:H*2, W:W*2, :] = input_map
        ys = int((H + ry) - half_size)
        ye = int((H + ry) - half_size + H)
        xs = int((W + rx) - half_size)
        xe = int((W + rx) - half_size + W)
        out_img = map_expanded[ys:ye, xs:xe, :]  # cropping back to the shifted org size
        return out_img

    def train_data_generator(self, world_idxs, num_rounds, shuffle=True, data_augment=True):
        random_world_idxs = np.asarray(world_idxs)
        random_round_idxs = np.arange(num_rounds)
        if shuffle:
            random.shuffle(random_world_idxs)
            random.shuffle(random_round_idxs)
        for widx in random_world_idxs:
            for roundidx in random_round_idxs:  # 99):
                curr_input_dir = '%s/%s/round%04d' % (self.dataset_dir, self.worlds[widx], roundidx)
                # count # of images
                num_data = len(fnmatch.filter(os.listdir(curr_input_dir), 'cmap_pad*'))
                random_data_idxs = np.arange(num_data)
                if shuffle:
                    random.shuffle(random_data_idxs)
                for ii in random_data_idxs:  # range(0, num_data):
                    #gtfile = '%s/rew_map_raw%04d.png' % (curr_input_dir, ii) #'%s/rew_map_raw%04d.png' % (curr_input_dir, ii)
                    #gt_rewmap_pad = self.load_padded_map(gtfile, self.pad_size, 0)  # 255 is at max coveraging spot
                    gtfile = '%s/rew_map_raw%04d.txt' % (curr_input_dir, ii) #'%s/rew_map_raw%04d.png' % (curr_input_dir, ii)
                    gt_rewmap_pad = self.load_padded_txt(gtfile, self.pad_size, 0, inv_map=False)  # relative raw coveraging scores.
                    metadata_file = '%s/processed_metadata%04d.txt' % (curr_input_dir, ii)
                    inputmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                    inputmap_pad = self.load_padded_map(inputmapfile, self.pad_size, 127)  # 127 is ukn in gridmap
                    frmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                    frmap_pad = self.load_padded_map(frmapfile, self.pad_size, 0)  # 255 is FR in FRimg
                    md = loadtxt(metadata_file)
                    (rx, ry, tx, ty) = md + self.pad_size
                    gt_rewmap_tform = self.transform_map_to_robotposition(gt_rewmap_pad, rx, ry, 0) # gtmap tform
                    inputmap_tform = self.transform_map_to_robotposition(inputmap_pad, rx, ry, 127) # gmap tform
                    frmap_tform = self.transform_map_to_robotposition(frmap_pad, rx, ry, 0) # frmap tform
                    input_tform = np.concatenate([inputmap_tform, frmap_tform], axis=-1)
                    input_tform = input_tform / 255.
                    max_rewmap_tform = np.max(gt_rewmap_tform)
                    gt_rewmap_tform = ( gt_rewmap_tform / max_rewmap_tform if max_rewmap_tform > 0 else gt_rewmap_tform )  #self.to_continous_label(gt_rewmap_pad, self.output_channels)
                    if data_augment:
                        input_img, gt_img = rotate(input_tform, gt_rewmap_tform)
                        input_img, gt_img = flip(input_img, gt_img)
                    yield input_img, gt_img  # shape: [B x H x W x C]

    def val_data_generator( self, widx, np_round_idxs ):
        for roundidx in np_round_idxs:  # 99):
            curr_input_dir = '%s/%s/round%04d' % (self.dataset_dir, self.worlds[widx], roundidx)
            # count # of images
            num_data = len(fnmatch.filter(os.listdir(curr_input_dir), 'cmap*'))
            for ii in range(0, num_data):
                print("validating %s round: %04d img idx: %04d \r" %(self.worlds[widx], roundidx, ii), end=' ')
                # gtfile = '%s/rew_map_raw%04d.png' % (curr_input_dir, ii) #'%s/rew_map_raw%04d.png' % (curr_input_dir, ii)
                # gt_rewmap_pad = self.load_padded_map(gtfile, self.pad_size, 0)  # 255 is at max coveraging spot
                gtfile = '%s/rew_map_raw%04d.txt' % (curr_input_dir, ii)  # '%s/rew_map_raw%04d.png' % (curr_input_dir, ii)
                gt_rewmap_pad = self.load_padded_txt(gtfile, self.pad_size, 0, inv_map=False)  # relative raw coveraging scores.
                metadata_file = '%s/processed_metadata%04d.txt' % (curr_input_dir, ii)
                inputmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                inputmap_pad = self.load_padded_map(inputmapfile, self.pad_size, 127)  # 127 is ukn in gridmap
                frmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                frmap_pad = self.load_padded_map(frmapfile, self.pad_size, 0)  # 255 is FR in FRimg
                md = loadtxt(metadata_file)
                (rx, ry, tx, ty) = md + self.pad_size
                gt_rewmap_tform = self.transform_map_to_robotposition(gt_rewmap_pad, rx, ry, 0)  # gtmap tform
                inputmap_tform = self.transform_map_to_robotposition(inputmap_pad, rx, ry, 127)  # gmap tform
                frmap_tform = self.transform_map_to_robotposition(frmap_pad, rx, ry, 0)  # frmap tform
                input_tform = np.concatenate([inputmap_tform, frmap_tform], axis=-1)
                input_tform = input_tform / 255.
                max_rewmap_tform = np.max(gt_rewmap_tform)
                gt_rewmap_tform = ( gt_rewmap_tform / max_rewmap_tform if max_rewmap_tform > 0 else gt_rewmap_tform )  #gt_rewmap_tform = gt_rewmap_tform / 255. # self.to_continous_label(gt_rewmap_pad, self.output_channels)
                yield input_tform[np.newaxis, :, :], gt_rewmap_tform[np.newaxis, :, :]  # shape: [B x H x W x C]
    def test_data_generator(self, widx, np_round_idxs ):
        for roundidx in np_round_idxs:  # 99):
            curr_input_dir = '%s/%s/round%04d' % (self.dataset_dir, self.worlds[widx], roundidx)
            # count # of images
            num_data = len(fnmatch.filter(os.listdir(curr_input_dir), 'cmap*'))
            for ii in range(0, num_data):
                #print("\t testing %s round: %04d img idx: %04d \r" %(self.worlds[widx], roundidx, ii), end=' ' )
                # gtfile = '%s/rew_map_raw%04d.png' % (curr_input_dir, ii) #'%s/rew_map_raw%04d.png' % (curr_input_dir, ii)
                # gt_rewmap_pad = self.load_padded_map(gtfile, self.pad_size, 0)  # 255 is at max coveraging spot
                gtfile = '%s/rew_map_raw%04d.txt' % (curr_input_dir, ii)  # '%s/rew_map_raw%04d.png' % (curr_input_dir, ii)
                gt_rewmap_pad = self.load_padded_txt(gtfile, self.pad_size, 0, inv_map=False)  # relative raw coveraging scores.
                metadata_file = '%s/processed_metadata%04d.txt' % (curr_input_dir, ii)
                inputmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                inputmap_pad = self.load_padded_map(inputmapfile, self.pad_size, 127)  # 127 is ukn in gridmap
                frmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                frmap_pad = self.load_padded_map(frmapfile, self.pad_size, 0)  # 255 is FR in FRimg
                md = loadtxt(metadata_file)
                (rx, ry, tx, ty) = md + self.pad_size
                gt_rewmap_tform = self.transform_map_to_robotposition(gt_rewmap_pad, rx, ry, 0)  # gtmap tform
                inputmap_tform = self.transform_map_to_robotposition(inputmap_pad, rx, ry, 127)  # gmap tform
                frmap_tform = self.transform_map_to_robotposition(frmap_pad, rx, ry, 0)  # frmap tform
                input_tform = np.concatenate([inputmap_tform, frmap_tform], axis=-1)
                input_tform = input_tform / 255.
                max_rewmap_tform = np.max(gt_rewmap_tform)
                gt_rewmap_tform = (gt_rewmap_tform / max_rewmap_tform if max_rewmap_tform > 0 else gt_rewmap_tform)
                #gt_rewmap_tform = gt_rewmap_tform / 255. # self.to_continous_label(gt_rewmap_pad, self.output_channels)
                yield input_tform[np.newaxis, :, :], gt_rewmap_tform[np.newaxis, :, :]  # shape: [B x H x W x C]
