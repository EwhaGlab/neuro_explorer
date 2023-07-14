
clear all; close all; clc
respath = '/home/hankm/results/neuro_ffp';
cnt = 0;

worlds = {'room', 'corner', 'corridor', 'loop_with_corridor'}
ridx = 99 ;
widx = round(3*rand) + 1 ; 

for idx=0:18
    
    imgfile = sprintf('%s/%s/round%04d/mapimg%04d.png',respath, worlds{widx}, ridx, idx);
    ffpfile = sprintf('%s/%s/round%04d/frimg%04d.png',respath, worlds{widx}, ridx, idx);
    datfile = sprintf('%s/%s/round%04d/metadata%04d.txt',respath, worlds{widx}, ridx, idx);

    I_map = imread(imgfile) ;
    I_ffp = imread(ffpfile) ;
    
%   check the CPP code
% 	rx ry qx qy qz qw tx ty gm_height gm_width gm_ox gm_oy resolution OFFSET roi.x roi.y roi.height roi.width 
    stdata = loadmetadata( datfile ) ;
    [v_fr u_fr] = find(I_ffp) ;

    figure(1)
    imshow(imgfile) ;
    hold on
    plot(u_fr + stdata.roi_x - stdata.offset, v_fr + stdata.roi_y - stdata.offset, 'r.' ) ;
    plot(stdata.rx + stdata.roi_x, stdata.ry + stdata.roi_y, 'c+', 'markersize', 10) ;
    plot(stdata.tx + stdata.roi_x, stdata.ty + stdata.roi_y, 'ms', 'MarkerFaceColor', 'y', 'markersize', 10) ;

    pause

    exportgraphics(h,sprintf('~/Desktop/Weekly/2023/Jun/data%04d.jpg',idx),'Resolution',300)
end



for idx=0:99

    % generate gt folder
    
    cmd = sprintf('/home/hankm/results/neuro_ffp/%s/round%04d/mapimg* /home/hankm/data/neuro_ffp/%s/ ') ;

    I_map = imread(imgfile) ;
    I_ffp = imread(ffpfile) ;
    
%   check the CPP code
% 	rx ry qx qy qz qw tx ty gm_height gm_width gm_ox gm_oy resolution OFFSET roi.x roi.y roi.height roi.width 
    stdata = loadmetadata( datfile ) ;
    [v_fr u_fr] = find(I_ffp) ;

    figure(1)
    imshow(imgfile) ;
    hold on
    plot(u_fr + stdata.roi_x - stdata.offset, v_fr + stdata.roi_y - stdata.offset, 'r.' ) ;
    plot(stdata.rx + stdata.roi_x, stdata.ry + stdata.roi_y, 'c+', 'markersize', 10) ;
    plot(stdata.tx + stdata.roi_x, stdata.ty + stdata.roi_y, 'ms', 'MarkerFaceColor', 'y', 'markersize', 10) ;

    pause

    exportgraphics(h,sprintf('~/Desktop/Weekly/2023/Jun/data%04d.jpg',idx),'Resolution',300)
end
