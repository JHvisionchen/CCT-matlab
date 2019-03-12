function [rects] = detect_object(im, weights, bias, ...
	object_sz, features, detection, threshold)
%  detect some candidates more similar to the object 
%  Guibo Zhu, 2015

	%activate to visualize response maps at each scale. 
    %%origin_im = im;  %% for debugging and displaying
	im = single(im);  
    im = double(im)./255;
	%object size as a fraction of the image size
	scale_factor = 1/features.detect_object_scale;
    
    scale_model_sz = size(im)*scale_factor;
     % resize image to model size
    if scale_factor~=1
       im = mexResize(im, scale_model_sz, 'auto');
    end
	
	
	%always reduce image, to detect at larger template sizes. start with
	%original image, to detect the smallest objects, and reduce it to
	%detect increasingly larger objects.
	rects = [];

       z = im-0.5;
       cell_size = 1;
		y = zeros(size(z,1), size(z,2), 'single');
		y(:) = bias;  %add bias term
       
		for f = 1:size(weights,3),
			y = y + imfilter(z(:,:,f), weights(:,:,f),'corr');     
        end
		
		%suppress responses that touch the borders
		y(1 : floor(size(weights,1)/2), :) = -inf;
		y(end - floor(size(weights,1)/2) + 1 : end, :) = -inf;
		y(:, 1 : floor(size(weights,2)/2)) = -inf;
		y(:, end - floor(size(weights,2)/2) + 1 : end) = -inf;

		%find response peaks, obtaining a set of rectangles for this scale
		scale_rects = get_rects(y, detection.max_peaks, threshold, ...
			object_sz / cell_size, detection.suppressed_scale);
		
		%resize and add them to the list. rescaling coordinates must be
		%done with respect to the origin (0-based instead of 1-based).
		scale_rects(:,1:2) = scale_rects(:,1:2) - 1;
		scale_rects(:,1:4) = scale_rects(:,1:4) * cell_size / scale_factor;
		scale_rects(:,1:2) = scale_rects(:,1:2) + 1;
		rects = [rects; scale_rects];  
	
	%perform non-maximum suppression
	rects = post_process_rects(rects, detection.max_overlap);
    
%     if 1
%         figure;imshow(origin_im);
%        % figure('Number','off', 'Name',['Tracker - ' video_path]);
%         for id = size(rects,1):-1:1
%             rectangle('Position',rects(id,1:4), 'EdgeColor','r', 'LineWidth',3);
%             %rectangle('Position',rects(id,1:4), 'EdgeColor','r');
%         end
%     end
end

