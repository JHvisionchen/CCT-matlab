% compute_pce_plane.m
%	* This function measures the peak sharpness of the correlation output.
%   * It computes a metric known as peak-to-correlation energy ratio.
%	* When comparing scores across image scales you will need calibration. 
%   *
%	* Created by Vishnu Naresh Boddeti on 5/22/13.
%	* naresh@cmu.edu (http://www.cs.cmu.edu/~vboddeti)
%	* Copyright 2013 Carnegie Mellon University. All rights reserved.

function [x,y,psr] = compute_pce_plane(corrplane,target_sz)

corrplane_sm=corrplane(round(size(corrplane,1)/2-target_sz(1)/4):round(size(corrplane,1)/2+target_sz(1)/4),round(size(corrplane,2)/2-target_sz(2)/4):round(size(corrplane,2)/2+target_sz(2)/4));
%max_c=max(corrplane_sm(:));
max_c = max(corrplane(:));
[y,x] = find(corrplane == max_c);
%[y,x] = find(corrplane_sm==max_c);
num_b=1;
if size(corrplane_sm,1)>11 && size(corrplane_sm,2)>11
     for i=1:size(corrplane_sm,1)
         for j=1:size(corrplane_sm,2)
             if ~(i>y-5&&i<y+5&&j>x-5&&j<x+5)
             temp(num_b)=corrplane_sm(i,j);num_b=num_b+1;
             end
         end
     end
     std_c=std(temp(:));
     mean_c=mean(temp(:));
     psr=(max_c-mean_c)/std_c;
else
%     comp_cor_sm=corrplane(a,b);
%     corrplane_sm=comp_cor_sm((size(comp_cor_sm,1)/2-target_sz(1)/2):(size(comp_cor_sm,1)/2+target_sz(1)/2),(size(comp_cor_sm,2)/2-target_sz(2)/2):(size(comp_cor_sm,2)/2+target_sz(2)/2));
%     std_c=std(corrplane_sm(:));
%     mean_c=mean(corrplane_sm(:));
     psr = 50;
end
 
% comp_cor_sm=corrplane(a,b);
% corrplane_sm=comp_cor_sm((size(comp_cor_sm,1)/2-target_sz(1)/2):(size(comp_cor_sm,1)/2+target_sz(1)/2),(size(comp_cor_sm,2)/2-target_sz(2)/2):(size(comp_cor_sm,2)/2+target_sz(2)/2));
% std_c=std(corrplane_sm(:));
% mean_c=mean(corrplane_sm(:));
% std_c=std(corrplane_sm(:));
% mean_c=mean(corrplane_sm(:));

%psr = 1/(std_c*sqrt(2*pi))*exp(-(max_c - mean_c)^2/(2*std_c^2));
%psr = (max_c - mean_c)/std_c/sqrt(prod(target_sz));
%corrplane = normalize_image(corrplane);
% startpoint=[floor(size(corrplane,1)/2)-20,floor(size(corrplane,2)/2)-20];
% pce_t=corrplane(startpoint(1):startpoint(1)+40,startpoint(2):startpoint(2)+40);

