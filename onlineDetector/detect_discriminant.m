function [rects] = detect_discriminant(im, weights, bias, ...
	object_sz, cell_size, features, detection, threshold)
%DETECT
%   Detects objects in a given image, using a single linear template.
%
%   RECTS = DETECT(IM, WEIGHTS, BIAS, OBJECT_SZ, CELL_SIZE, FEATURES,
%     DETECTION, THRESHOLD)
%   Detects objects in image IM, with template WEIGHTS and the classifier
%   offset BIAS. Only detections with score above THRESHOLD are returned.
%   OBJECT_SZ contains the dimensions of the object relative to the
%   template size (predicted bounding box size). CELL_SIZE is the cell
%   size of the dense features, which are specified in struct FEATURES
%   (see function GET_FEATURES).
%
%   The DETECTION struct contains the following fields:
%   'min_scale' and 'max_scale': Min./max. scale, relative to image size.
%   'scale_step': Scale factor applied to get the next scale (> 1).
%   'max_peaks': Maximum number of detections per scale.
%   'suppressed_scale': Response suppression around peaks, relative to the
%     object size.
%   'max_overlap': Maximum relative area overlap between two detections.
%
%   Joao F. Henriques, 2013


	%activate to visualize response maps at each scale. calls "pause", so
	%it is most useful for debugging the detector on a single image.
	%debug_scales = true;

	im = single(im);  %without this, AP decreases a bit (explanations welcome!)
    
	%object size as a fraction of the image size
	scale_factor = 1/features.detect_object_scale;
    %im = imresize(im, scale_factor, 'bilinear');
    
    scale_model_sz = size(im)*scale_factor;
     % resize image to model size
    im = mexResize(im, scale_model_sz, 'auto');
	
% 	%first, reduce image so that the smallest detected object has the size
% 	%of the template
% 	if obj_scale <= detection.min_scale,
% 		scale_factor = obj_scale / detection.min_scale;
% 		im = imresize(im, scale_factor, 'bilinear');
% 		obj_scale = detection.min_scale;
% 	end  %else, smaller scales may be skipped (the template is too big)
	
	
	%always reduce image, to detect at larger template sizes. start with
	%original image, to detect the smallest objects, and reduce it to
	%detect increasingly larger objects.
	rects = [];

% 	while obj_scale < detection.max_scale,
		%note: we only resize the image at the end, so that the first
		%iteration corresponds to the original, non-rescaled size.

		%extract features (e.g., HOG)
       % t2 = 0;
        tic;
		z = get_features(im, features, cell_size, features.hog_orientations);
		%t2 = toc;
		%convolve with template to obtain response (same width/height as z)
        %t3 = 0;
        tic;
		y = zeros(size(z,1), size(z,2), 'single');
		y(:) = bias;  %add bias term
		for f = 1:size(weights,3),
			y = y + imfilter(z(:,:,f), weights(:,:,f));
        end
        %t3=toc;
		
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

% 		if debug_scales,  %show plot of response map at current scale
% 			figure(1), set(gcf, 'Number','off', 'Name', ['Scale: ', num2str(obj_scale)])
% 			imagesc(resized_y)  %might also want to visualize "scale_rects" (not implemented)
% 			pause
% 		end

% 		%resize image
% 		obj_scale = obj_scale * detection.scale_step;
% 		if obj_scale < detection.max_scale,  %don't waste time resizing in the last iteration
% 			scale_factor = scale_factor / detection.scale_step;
% 			im = imresize(im, 1 / detection.scale_step, 'bilinear');
% 		end
	%end
	
	%perform non-maximum suppression
	rects = post_process_rects(rects, detection.max_overlap);
    %rects = post_process_rects(rects, 0.6);
    
   %%get the corresponding negative samples' features
   weight_sz = size(weights);
   rect_features = zeros(weight_sz(1), weight_sz(2), weight_sz(3),size(rects,1));
   feat_rects = rects*scale_factor/cell_size;
  
    if 0
        %imshow(im);
       % figure('Number','off', 'Name',['Tracker - ' video_path]);
        for id = size(rects,1):-1:1
            rectangle('Position',rects(id,1:4), 'EdgeColor','b');
        end
    end
% 	rects = post_process_rects(rects, detection.max_overlap, [1, 1, im_sz(2)-1, im_sz(1)-1]);
end

