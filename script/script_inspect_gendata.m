
clear all; close all; clc

respath = '/home/hankm/results/cnn_map_reward';
cnt = 0;

for idx=0:99
    
    imgfile_prev = sprintf('%s/map_before_moving%05d.pgm',respath,idx);
    datfile_prev  = sprintf('%s/metadata_before_moving%05d.dat',respath,idx);
    datfile_next  = sprintf('%s/metadata_after_moving%05d.dat',respath,idx);
    freesfile= sprintf('%s/frees%05d.csv',respath,idx);
    imgfile_next = sprintf('%s/map_after_moving%05d.pgm',respath,idx);
    
    I_prev = imread(imgfile_prev) ;
    I_next = imread(imgfile_next) ;
    metadata_prev = load(datfile_prev);
    !freesidx = load(freesfile);
    xp0 = metadata_prev(1);
    yp0 = metadata_prev(2);
    xn0 = metadata_prev(3);
    yn0 = metadata_prev(4);

    metadata_next = load(datfile_next);
    xp1 = metadata_next(1);
    yp1 = metadata_next(2);
    xn1 = metadata_next(3);
    yn1 = metadata_next(4);
    rew = metadata_next(5)

    cov_prev = (I_prev ~= 127 ) ;
    cov_next = (I_next ~= 127 ) ;
    [sum(cov_next(:))*0.05*0.05 - sum(cov_prev(:))*0.05*0.05 rew]

    [xp0 yp0 xp1 yp1]

    h1=figure(cnt+1); clf
    subplot(1,2,1)
    title(sprintf('dataidx= %d init map\n',idx))
    hold on
    imshow(I_prev);
    plot(xp0, yp0, 'c+', 'markersize', 12) ;
    plot(xn0, yn0, 'ys', 'markersize', 12) ;
    
    h1; 
    subplot(1,2,2); 
    title(sprintf('dataidx= %d after moving the robot\n',idx))
    hold on
    imshow(I_next);
    !plot(freesidx(2,:), freesidx(1,:), 'r.', 'markersize', 1 )
    plot(xn1, yn1, 'c+', 'markersize', 12) ; % moved robot position
    pause

    cnt = mod(cnt+1,2);
    idx
end










respath = '/home/hankm/results/cnn_map_reward';

for idx=0:99
    
    imgfile = sprintf('%s/map%05d.pgm',respath,idx);
    datfile = sprintf('%s/metadata%05d.dat',respath,idx);
    freesfile =sprintf('%s/frees%05d.csv',respath,idx);

    I = imread(imgfile)  ;
    metadata = load(datfile);
    freesidx = load(freesfile);
    xi = metadata(1);
    yi = metadata(2);
    xr = metadata(3);
    yr = metadata(4);

    figure(idx+1)
    imshow(I);
    hold on
    plot(freesidx(2,:), freesidx(1,:), 'r.' )
    plot(xi, yi, 'c+') ;
    plot(xr, yr, 'bs') ;

    pause

end

