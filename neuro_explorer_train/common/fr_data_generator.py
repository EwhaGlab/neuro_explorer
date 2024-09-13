
import sys
import os
import keras.metrics

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import fnmatch
import random
base_dir = os.path.abspath('../../')
sys.path.insert(0, '%s/utils'%base_dir)
from data_augmentation import rotate, flip
from numpy import loadtxt
from dataclasses import dataclass

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

class fr_data_generator():
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
    # inclusive label causes loss jumping ( up to ~20K ). Not sure what is the reason. Thus, this function is not recommended..
    def to_mutal_inclusive_label( self, gtimg ): # gtimg uint8 w/ max 255
        assert(gtimg.ndim == 3)
        nonfr = np.where( gtimg == 0, 1, 0) # 0 means non FR
        fr = np.where( gtimg > 0, 1, 0) # 127 and 255 are both  FR
        optfr = np.where( gtimg == 2, 1, 0) # 255 is opt FR
        gt_class = np.concatenate( (nonfr, fr, optfr), axis=2 ).astype(np.float32)
        return gt_class
    def to_mutal_exclusive_label( self, gtimg ): # gtimg uint8 w/ max 255
        assert( gtimg.ndim == 3) # we need H X W X C
        nonfr = np.where( gtimg == 0, 1, 0) # 0 means non FR
        fr = np.where( gtimg == 1, 1, 0) # 127 and 255 are both  FR
        optfr = np.where( gtimg == 2, 1, 0) # 255 is opt FR
        (r,c,d) = np.where(fr * optfr > 0) # where both fr and optfr
        nonfr[r, c] = 0
        fr[r,c] = 0
        optfr[r,c] = 1
        gt_class = np.concatenate( (nonfr, fr, optfr), axis=2 ).astype(np.float32)
        gt_class
        return gt_class
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
                    # print("%s round: %d img idx: %d\n"%(worlds[widx],ridx,ii) )
                    # cmapfile = '%s/cmap%04d.png' % (curr_input_dir, ii)
                    gtmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                    gmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                    gmap_pad = self.load_padded_map(gmapfile, self.pad_size, 127) # 0, 127, 255
                    gtmap_pad = self.load_padded_map(gtmapfile, self.pad_size, 0) # 0, 255
                    input_img = gmap_pad / 255.
                    gt_img = gtmap_pad / 255.
                    if data_augment:
                        input_img, gt_class = rotate(input_img, gt_img)
                        input_img, gt_class = flip(input_img, gt_img)
                    yield input_img, gt_class  # binary class, shape: [B x H x W x C]
    def val_data_generator( self, widx, np_round_idxs):
        for roundidx in np_round_idxs:  # 99):
            curr_input_dir = '%s/%s/round%04d' % (self.dataset_dir, self.worlds[widx], roundidx)
            #print("curr val input dir: ", curr_input_dir)
            # count # of images
            num_data = len(fnmatch.filter(os.listdir(curr_input_dir), 'cmap*'))
            for ii in range(0, num_data):
                #print("validating %s round: %d img idx: %d\n" %(self.worlds[widx], roundidx, ii) )
                # cmapfile = '%s/cmap%04d.png' % (curr_input_dir, ii)
                # print("%s round: %d img idx: %d\n"%(worlds[widx],ridx,ii) )
                # cmapfile = '%s/cmap%04d.png' % (curr_input_dir, ii)
                gtmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                gmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                gmap_pad = self.load_padded_map(gmapfile, self.pad_size, 127) # 0, 127, 255
                gtmap_pad = self.load_padded_map(gtmapfile, self.pad_size, 0) # 0, 255
                input_img = gmap_pad / 255.
                gt_class = gtmap_pad / 255.
                yield input_img[np.newaxis,...], gt_class[np.newaxis,...]  # binary class, shape: [B x H x W x C]
                #yield input_img, gt_class  # binary class, shape: [B x H x W x C]

    def test_data_generator( self, widx, np_round_idxs ):
        for roundidx in np_round_idxs:  # 99):
            curr_input_dir = '%s/%s/round%04d' % (self.dataset_dir, self.worlds[widx], roundidx)
            #print("curr val input dir: ", curr_input_dir)
            # count # of images
            num_data = len(fnmatch.filter(os.listdir(curr_input_dir), 'cmap*'))
            for ii in range(0, num_data):
                #print("validating %s round: %d img idx: %d\n" %(self.worlds[widx], roundidx, ii) )
                # cmapfile = '%s/cmap%04d.png' % (curr_input_dir, ii)
                # print("%s round: %d img idx: %d\n"%(worlds[widx],ridx,ii) )
                # cmapfile = '%s/cmap%04d.png' % (curr_input_dir, ii)
                gtmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                gmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                gmap_pad = self.load_padded_map(gmapfile, self.pad_size, 127) # 0, 127, 255
                gtmap_pad = self.load_padded_map(gtmapfile, self.pad_size, 0) # 0, 255
                input_img = gmap_pad / 255.
                gt_class = gtmap_pad / 255.
                yield input_img[np.newaxis,...], gt_class[np.newaxis,...]  # binary class, shape: [B x H x W x C]
