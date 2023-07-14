function stdata = loadmetadata( datfile )

%   check the CPP code
%   1   2   3   4   5   6   7   8   9           10          11      12      13          14      15      16      17          18                                          
% 	rx  ry  qx  qy  qz  qw  tx  ty  gm_height   gm_width    gm_ox   gm_oy   resolution  OFFSET  roi.x   roi.y   roi.height  roi.width 

    metadata = load(datfile);
    rx_w    = metadata(1) ;
    ry_w    = metadata(2) ;
    tx_w    = metadata(7) ;
    ty_w    = metadata(8) ;
    height  = metadata(9) ;
    width   = metadata(10) ;
    ox      = metadata(11) ;
    oy      = metadata(12) ;
    res     = metadata(13);
    offset  = metadata(14) ;
    roi_x   = metadata(15) ;
    roi_y   = metadata(16) ;
    roi_height = metadata(17) ;
    roi_width  = metadata(18) ;

    stdata = struct 
    stdata.rx = ( rx_w - ox ) / res ;
    stdata.ry = ( ry_w - oy ) / res ;
    stdata.tx = ( tx_w - ox ) / res ;
    stdata.ty = ( ty_w - oy ) / res ;
    stdata.gmheight = height ;
    stdata.gmwidth  = width ;
    stdata.roi_x = roi_x ;
    stdata.roi_y = roi_y ;
    stdata.roi_height = roi_height ;
    stdata.roi_width = roi_width ;
    stdata.offset = offset ;
end
