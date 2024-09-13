
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import cv2
import fnmatch
import random
sys.path.insert(0, '/home/hankm/python_ws/neuro_ae/utils')
from data_augmentation import rotate, flip
from numpy import loadtxt
from dataclasses import dataclass
import math

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
        self.fr_size_thr        = int( kwargs['data_configs']['fr_size_thr'])
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

    def load_padded_txt(self, potmapfile, pad_size, base_val, inv_map=True):
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
    def load_potmap_on_canv( self, potmapfile, metadata, canvsize = 1000 ): # need to check if the mapping is correct
        img_raw = loadtxt(potmapfile)       # potmap -1, 0 ~ maxpot
        maxpot = np.max(img_raw)
        if maxpot == 0: # meaning there is no FR pt to compute Astar dist(pot)
            img_ = img_raw
        else:
            img_ = ( img_raw / np.max(img_raw) ) #* 247          # - , 0 ~ 247 (max pot)
        mappot_inv = np.where( img_raw > 0, (1. - img_ ) * 247, 0)   #     1 (max pot) ~ 247 (min pot), 0 (FREE <non-FR> or OBS)
        img_canv = np.zeros([canvsize, canvsize], dtype=np.float32)
        sx = int( metadata.roi_x - metadata.offset )
        ex = int( sx + metadata.gmwidth + metadata.offset * 2)
        sy = int( metadata.roi_y - metadata.offset )
        ey = int( sy + metadata.gmheight + metadata.offset * 2)
        img_canv[sy:ey, sx:ex] = mappot_inv
        return img_canv[..., np.newaxis]
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
    # inclusive label causes loss jumping ( up to ~20K ). Not sure what is the reason. Thus, this function is not recommended..
    def potmap_to_label( self, gtimg_input ): # gtimg uint8 w/ max 255
        gtimg = gtimg_input.squeeze()
        (h, w) = gtimg.shape
        num_classes = self.output_channels
        #tot_steps = num_classes - 1 # 1 ~ 255 is quantified into 30 bins then the last extra two channels are added for -255(non FR cells) and zeros
        gt_class = np.zeros([h, w, num_classes], dtype=np.float32)
        step_size = int( 256 / num_classes )
        free_bin = np.where(gtimg <= 0, 1, 0) # free cells
        gt_class[:, :, 0] = free_bin.squeeze()
        for step_idx in range(0, num_classes):
            m_floor = np.where(gtimg > (step_idx) * step_size, 1, 0)
            m_ceil  = np.where(gtimg <= (step_idx+1) * step_size, 1, 0)
            m_bin = np.multiply(m_ceil, m_floor)
            gt_class[:, :, step_idx] = m_bin.squeeze().astype(np.float32)
        return gt_class
    def label_to_potmap_discrete( self, in_class ): # gtimg uint8 w/ max 255
        assert(in_class.ndim == 3)
        num_classes = self.output_channels
        bimg = np.argmax(in_class, axis=-1) * 255 / (num_classes-1)  # 0 ~ 31
        #np.where(bimg == num_classes-1, -1, bimg * 255 )
        return bimg.astype(np.uint8)
    def binlabl_to_bimg( self, blabel, num_classes ):
        bimg = np.where(blabel > 0.5, 1, 0)
        return bimg
    def labels_to_bimg( self, in_class, num_classes ): # gtimg uint8 w/ max 255
        assert(in_class.ndim == 3)
        bimg = np.argmax(in_class, axis=-1) / (num_classes-1) * 255
        return bimg.astype(np.uint8)
    def preprocess_inputmap(self, fnp_inmap, thr=5):
        '''
        :param fnp_inmap: input map (float32 type)
        :param thr: cluster size thr
        :return: noisy cluster filtered map
        '''
        fnp_inmap_bin = np.where(fnp_inmap > 0, 1, 0).astype('uint8')
        contours, hierarchy = cv2.findContours(fnp_inmap_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (r, c) = fnp_inmap.squeeze().shape
        fnp_mask_flat = np.zeros( [r*c], dtype=np.float32)
        for idx in range(len(contours)):
            subidxs  = contours[idx].squeeze()
            cl_size = subidxs.shape[0]
            if(cl_size > thr):
                ind = subidxs[:, 0]*r + subidxs[:, 1]
                fnp_mask_flat[ind] = 1.0
        fnp_mask = np.reshape(fnp_mask_flat, (r, c), 'F')
        fnp_mask_ch = fnp_mask[..., np.newaxis]
        fnp_masked = fnp_inmap * fnp_mask_ch
        maxval = np.max(fnp_masked)
        if (maxval > 0):
            return fnp_masked / maxval
        else:
            return fnp_masked
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
                    gttxtfile = '%s/processed_potmap%04d.txt' % (curr_input_dir, ii)
                    metadata_file = '%s/processed_metadata%04d.txt' % (curr_input_dir, ii)
                    gmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                    gmap_pad = self.load_padded_map(gmapfile, self.pad_size, 0)
                    obs_pad = np.where(gmap_pad == 255, 1, 0)
                    rpose_map = np.zeros([self.gmheight_p, self.gmwidth_p], dtype=np.float32)
                    md = loadtxt(metadata_file)
                    gt_potmap_pad = self.load_padded_txt(gttxtfile, self.pad_size, 0, True) #* 255.  # 1.0 is at max pot (including obs)
                    inputmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                    inputmap_pad = self.load_padded_map(inputmapfile, self.pad_size, 0)  # 255 is FR in FRimg
                    ##################### filter out noisy pixels of both GT and input #############################
                  #  inputmap_pad = self.preprocess_inputmap(inputmap_pad, self.fr_size_thr)
                  #  gt_potmap_pad = self.preprocess_inputmap(gt_potmap_pad, self.fr_size_thr)
                    ################################################################################################
                    (rx, ry, tx, ty) = md + self.pad_size
                    gt_potmap_tform = self.transform_map_to_robotposition(gt_potmap_pad, rx, ry, 0)
                    rpose_gauss_pad_tform = self.add_gaussian_to_pixel(rpose_map, self.gmheight_p / 2, self.gmwidth_p / 2, 0.5,
                                                                  kernel_size=127)  # rpose gaussing map
                    input_map_tform = self.transform_map_to_robotposition(inputmap_pad, rx, ry, 0) / 255.  # fr img map
                    obs_map_tform = self.transform_map_to_robotposition(obs_pad, rx, ry, 0)  # obs map
                    input_img = np.concatenate([input_map_tform, obs_map_tform, rpose_gauss_pad_tform],
                                               axis=-1)  # input_map_tform / 255.
                    gt_class = gt_potmap_tform #self.to_continous_label(gt_potmap_tform, self.output_channels)
                    if data_augment:
                        input_img, gt_class = rotate(input_img, gt_class)
                        input_img, gt_class = flip(input_img, gt_class)
                    yield input_img, gt_class  # shape: [B x H x W x C]
    def val_data_generator( self, widx, np_round_idxs ):
        for roundidx in np_round_idxs:  # 99):
            curr_input_dir = '%s/%s/round%04d' % (self.dataset_dir, self.worlds[widx], roundidx)
            # count # of images
            num_data = len(fnmatch.filter(os.listdir(curr_input_dir), 'cmap*'))
            for ii in range(0, num_data):
                print("validating %s round: %04d img idx: %04d \r" %(self.worlds[widx], roundidx, ii), end=' ' )
                # cmapfile = '%s/cmap%04d.png' % (curr_input_dir, ii)
                gttxtfile = '%s/processed_potmap%04d.txt' % (curr_input_dir, ii)
                metadata_file = '%s/processed_metadata%04d.txt' % (curr_input_dir, ii)
                gmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                gmap_pad = self.load_padded_map(gmapfile, self.pad_size, 0)
                obs_pad = np.where(gmap_pad == 255, 1, 0)
                rpose_map = np.zeros([self.gmheight_p, self.gmwidth_p], dtype=np.float32)
                md = loadtxt(metadata_file)
                gt_potmap_pad = self.load_padded_txt(gttxtfile, self.pad_size, 0, True) #* 255.  # 1.0 is at max pot (including obs)
                inputmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                inputmap_pad = self.load_padded_map(inputmapfile, self.pad_size, 0)  # 255 is FR in FRimg
                ##################### filter out noisy pixels of both GT and input #############################
             #   inputmap_pad = self.preprocess_inputmap(inputmap_pad, self.fr_size_thr)
             #   gt_potmap_pad = self.preprocess_inputmap(gt_potmap_pad, self.fr_size_thr)
                ################################################################################################
                (rx, ry, tx, ty) = md + self.pad_size
                gt_potmap_tform = self.transform_map_to_robotposition(gt_potmap_pad, rx, ry, 0)
                rpose_gauss_pad_tform = self.add_gaussian_to_pixel(rpose_map, self.gmwidth_p / 2, self.gmwidth_p / 2, 0.5, kernel_size=127)  # rpose gaussing map
                input_map_tform = self.transform_map_to_robotposition(inputmap_pad, rx, ry, 0) / 255.  # fr img map
                obs_map_tform = self.transform_map_to_robotposition(obs_pad, rx, ry, 0)  # obs map
                input_img = np.concatenate([input_map_tform, obs_map_tform, rpose_gauss_pad_tform], axis=-1)  # input_map_tform / 255.
                gt_class = gt_potmap_tform #self.to_continous_label(gt_potmap_tform, self.output_channels)
                #yield input_img[np.newaxis, ...], gt_class[np.newaxis, ...]  # shape: [B x H x W x C] # use it for test data generator
                #print("input_img shape: ", input_img.shape)
                #print("gt class shape: ", gt_class.shape)
                yield input_img[np.newaxis, ...], gt_class[np.newaxis, ...] # shape: [H x W x C]
    def test_data_generator(self, widx, np_round_idxs ):
        for roundidx in np_round_idxs:  # 99):
            curr_input_dir = '%s/%s/round%04d' % (self.dataset_dir, self.worlds[widx], roundidx)
            # count # of images
            num_data = len(fnmatch.filter(os.listdir(curr_input_dir), 'cmap*'))
            for ii in range(0, num_data):
                # print("%s round: %d img idx: %d\n"%(worlds[widx],ridx,ii) )
                # cmapfile = '%s/cmap%04d.png' % (curr_input_dir, ii)
                gttxtfile = '%s/processed_potmap%04d.txt' % (curr_input_dir, ii)
                metadata_file = '%s/processed_metadata%04d.txt' % (curr_input_dir, ii)
                gmapfile = '%s/processed_gmap%04d.png' % (curr_input_dir, ii)
                gmap_pad = self.load_padded_map(gmapfile, self.pad_size, 0)
                obs_pad = np.where(gmap_pad == 255, 1, 0)
                rpose_map = np.zeros([self.gmheight_p, self.gmwidth_p], dtype=np.float32)
                md = loadtxt(metadata_file)
                gt_potmap_pad = self.load_padded_txt(gttxtfile, self.pad_size, 0, True)  # 1.0 is at max pot (including obs)
                inputmapfile = '%s/processed_frimg%04d.png' % (curr_input_dir, ii)
                inputmap_pad = self.load_padded_map(inputmapfile, self.pad_size, 0)  # 255 is FR in FRimg
                ##################### filter out noisy pixels of both GT and input #############################
              #  inputmap_pad = self.preprocess_inputmap(inputmap_pad, self.fr_size_thr)
              #  gt_potmap_pad = self.preprocess_inputmap(gt_potmap_pad, self.fr_size_thr)
                ################################################################################################
                (rx, ry, tx, ty) = md + self.pad_size
                gt_potmap_tform = self.transform_map_to_robotposition(gt_potmap_pad, rx, ry, 0)
                rpose_gauss_pad_tform = self.add_gaussian_to_pixel(rpose_map, self.gmwidth_p / 2, self.gmwidth_p / 2, 0.5, kernel_size=127)  # rpose gaussing map
                input_map_tform = self.transform_map_to_robotposition(inputmap_pad, rx, ry, 0) / 255.  # fr img map
                obs_map_tform = self.transform_map_to_robotposition(obs_pad, rx, ry, 0)  # obs map
                input_img = np.concatenate([input_map_tform, obs_map_tform, rpose_gauss_pad_tform], axis=-1)  # input_map_tform / 255.
                gt_class = gt_potmap_tform #self.to_continous_label(gt_potmap_tform, self.output_channels)
                yield input_img[np.newaxis, ...], gt_class[np.newaxis, ...]  # shape: [B x H x W x C] # use it for test data generator
