function [positions, output_positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, params, show_visualization)

	%if the target is large, lower the resolution, we don't need that much
	%detail
    init_rect = round([pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])]);
    
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end
    if target_sz(1)<15 && target_sz(2) <15
        cell_size = 1;
    end
   
	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
    
	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
    use_object_filter =features.use_object_detector;
    Tn = 40;
    Tu = 0.05;
    if use_object_filter  %%whether using the CUR detection filter;
        %%detection parameters
        detection = [];
        detection.max_scale = 1;  %maximum scale, relative to image size
        detection.min_scale = 0.1;  %min. scale
        detection.scale_step = 1.1;  %relative scale step
        detection.max_peaks =10;  %maximum number of detections per scale
        detection.suppressed_scale = 0.8;  %response suppression around peaks, relative to object size
        detection.max_overlap = 0.1;  %maximum relative area overlap between two detections
        min_object_size = 32;
        
        threshold = -inf;
        detect_object_sz = target_sz;   
        detect_object_scale = 1.0;
        
        if min(target_sz)>min_object_size  %% setting the template object size;
           id = logical(target_sz==min(target_sz));
           detect_object_scale = target_sz(id)/min_object_size;
           detect_object_scale = detect_object_scale(1);
           detect_object_sz = target_sz/detect_object_scale;    
           
        end
       features.detect_object_scale = detect_object_scale;
       
       object_filter_pool = [];
       K = 20;   %%the value of c
       learning_rate = 0.025;  %%the learning rate of online CUR filter
    end
    is_occluded = false;
    is_add_pool = true;  %%whether adding the 
    
	%note: variables ending with 'f' are in the Fourier domain.
	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    output_positions = zeros(numel(img_files),4);
    
    prod_win_sz = prod(window_sz);

   use_scale_estimate = features.use_scale_estimate;
   if use_scale_estimate
        % target size att scale = 1
        base_target_sz = target_sz;
        init_target_sz = target_sz;
        num_scales = params.scale_range;           % number of scale levels (denoted "S" in the paper)
        scale_step = params.scale_step;               % Scale increment factor (denoted "a" in the paper)
        scale_model_max_area = 512;      % the maximum size of scale examples
        scale_sigma_factor = 1/4;
        learning_rate = 0.025;
        % desired scale filter output (gaussian shaped), bandwidth proportional to
        % number of scales
        scale_sigma = num_scales/sqrt(33) * scale_sigma_factor;
        ss = (1:num_scales) - ceil(num_scales/2);
        ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
        ysf = single(fft(ys));
        
        % store pre-computed scale filter cosine window
        if mod(num_scales,2) == 0
            scale_window = single(hann(num_scales+1));
            scale_window = scale_window(2:end);
        else
            scale_window = single(hann(num_scales));
        end;

        % scale factors
        ss = 1:num_scales;
        scaleFactors = scale_step.^(ceil(num_scales/2) - ss);

        % compute the resize dimensions used for feature extraction in the scale
        % estimation
        scale_model_factor = 1;
        if prod(init_target_sz) > scale_model_max_area
            scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
        end
        scale_model_sz = floor(init_target_sz * scale_model_factor);
        
        currentScaleFactor = 1;
        
        % find maximum and minimum scales
        im = imread([video_path img_files{1}]);
        min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
        max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
    end
    
	for frame = 1:numel(img_files),
		%load image
		img = imread([video_path img_files{frame}]);
        im=img;
        if features.use_rgb2gray
            if size(im,3) > 1,
              im = rgb2gray(im);
            end
        end   
		if resize_image,
			im = imresize(im, 0.5);
        end

		tic()
          
		if frame > 1,
			%obtain a subwindow for detection at the position from last
			%frame, and convert to Fourier domain (its size is unchanged)
			%patch = get_subwindow(im, pos, window_sz);
             if ~use_scale_estimate
                 patch = get_subwindow(im, pos, window_sz);
             else
                 patch = get_translation_subwindow(im, pos, window_sz, currentScaleFactor);
            end
			zf = fft2(get_features(patch, features, cell_size, cos_window));
        
			switch kernel.type
			case 'gaussian',
				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
			case 'polynomial',
				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
			case 'linear',
				kzf = linear_correlation(zf, model_xf);
			end
			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            
            
            %target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
            max_conf = max(response(:));     
			[vert_delta, horiz_delta] = find(response == max_conf);
            vert_delta = vert_delta(1);
            horiz_delta = horiz_delta(1);
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
            end
            pre_pos = pos; %%storing the last center position;
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1]*currentScaleFactor;
            %pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
            
     
            %the following code is to detect some confident object candidates using CUR filter;
            if use_object_filter,
                %%detect the confident candidates;
                rects = detect_object(im, object_filter,0,detect_object_sz,features, detection, threshold);
                cur_loc = [round(pos([2,1]) - target_sz([2,1])/2+0.0001), target_sz([2,1])];
                
                overlaps = compute_overlap(cur_loc,rects);
                index = find(overlaps==max(overlaps(:)));
                
                is_occluded = false;
                is_add_pool = true;
                
                if overlaps(index) < 0.5,
                    is_add_pool = false;%% whether adding the current object representation to the object pool;
                end
                if overlaps(index) < Tu & frame > Tn   %%whether adjusting the candidates for rectifying the estimated results;
                    is_occluded = true;
                    max_response = max_conf;
                    pos_set = pos;
                    for canId = 1:size(rects,1)
                        sal_pos = [rects(canId,2)+rects(canId,4)/2-1, rects(canId,1)+rects(canId,3)/2-1];
                        saliency_patch = get_subwindow(im,sal_pos, window_sz);
                        saliency_zf = fft2(get_features(saliency_patch, features, cell_size, cos_window));
                        %calculate response of the classifier at all shifts
                        saliency_kzf = gaussian_correlation(saliency_zf, model_xf, kernel.sigma);
                        saliency_response = real(ifft2(model_alphaf .* saliency_kzf));  %equation for fast detection

                        %target location is at the maximum response. we must take into
                        %account the fact that, if the target doesn't move, the peak
                        %will appear at the top-left corner, not at the center (this is
                        %discussed in the paper). the responses wrap around cyclically.
                        max_saliency_response = max(saliency_response(:));
                        [saliency_vert_delta, saliency_horiz_delta] = find(saliency_response == max_saliency_response);
                        if saliency_vert_delta > size(saliency_zf,1) / 2,  %wrap around to negative half-space of vertical axis
                            saliency_vert_delta = saliency_vert_delta - size(saliency_zf,1);
                        end
                        if saliency_horiz_delta > size(saliency_zf,2) / 2,  %same for horizontal axis
                            saliency_horiz_delta = saliency_horiz_delta - size(saliency_zf,2);
                        end
                        saliency_pos = sal_pos + cell_size * [saliency_vert_delta - 1, saliency_horiz_delta - 1];
                        max_response = [max_response;max_saliency_response];
                        pos_set = [pos_set;saliency_pos];                        
                    end
                    %%compute the spatial constraint;
                     num_pos_set = size(pos_set,1);
                     pos_weights = zeros(num_pos_set,1);
                     for posId = 1:num_pos_set,
                        cur_pos = pos_set(posId,:);
                        diff_cur_pre = (cur_pos(1)-pre_pos(1))^2+(cur_pos(2)-pre_pos(2))^2;
                        pos_weights(posId) = exp(-diff_cur_pre/prod_win_sz);
                     end
                    %%choose the most confident candidate as the object
                    %%state.
                     max_response = max_response.*pos_weights;
                     max_id = find(max_response ==max(max_response(:)),1);
                     pos = pos_set(max_id,:);
                end

        end
            
            
            if use_scale_estimate && frame>0.25*Tn
                % extract the test sample feature map for the scale filter
                xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);

                % calculate the correlation response of the scale filter
                xsf = fft(xs,[],2);
                scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + lambda)));

                % find the maximum scale response
                recovered_scale = find(scale_response == max(scale_response(:)), 1);

                % update the scale
                currentScaleFactor = currentScaleFactor * scaleFactors(recovered_scale);
                if currentScaleFactor < min_scale_factor
                    currentScaleFactor = min_scale_factor;
                elseif currentScaleFactor > max_scale_factor
                    currentScaleFactor = max_scale_factor;
                end
            end
            currentScaleFactor = currentScaleFactor(1); 
		end

		%obtain a subwindow for training at newly estimated target position
        if ~use_scale_estimate
             patch = get_subwindow(im, pos, window_sz);
        else
           patch = get_translation_subwindow(im, pos, window_sz, currentScaleFactor);
        end
        xf_feat = get_features(patch, features, cell_size, cos_window);
		xf = fft2(xf_feat);

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian',
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training
              
        if use_scale_estimate
            % extract the training sample feature map for the scale filter
            xs = get_scale_sample(im, pos, base_target_sz, currentScaleFactor * scaleFactors, scale_window, scale_model_sz);

            % calculate the scale filter update
            xsf = fft(xs,[],2);
            new_sf_num = bsxfun(@times, ysf, conj(xsf));
            new_sf_den = sum(xsf .* conj(xsf), 1);
        end

		if frame == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
            if use_scale_estimate
                sf_den = new_sf_den;
                sf_num = new_sf_num;
            end
            
            %%the initialization of the CUR filter;
            if use_object_filter 
                object_patch = get_subwindow(im, pos, target_sz);  
                object_patch =  mexResize(object_patch, detect_object_sz, 'auto');
                object_filter = double(object_patch)./255 - 0.5;  %subtract 
                object_filter_pool = object_filter;
            end

        else
           if ~is_occluded  %%whether updating the classifier;
               model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
               model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
           else
               model_alphaf = (1 - 0.1*interp_factor) * model_alphaf + 0.1*interp_factor * alphaf;
               model_xf = (1 - 0.1*interp_factor) * model_xf + 0.1*interp_factor * xf;
           end
            if use_scale_estimate && (~is_occluded)
                sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
                sf_num = (1 - learning_rate) * sf_num + learning_rate * new_sf_num;
            end
            
            %%update the CUR filter;
            if use_object_filter 
                object_patch = get_subwindow(im, pos, target_sz);
                object_patch =  mexResize(object_patch, detect_object_sz, 'auto');
                tmp_object_filter = double(object_patch)./255 - 0.5;
                if is_add_pool
                    object_filter_pool = cat(3, object_filter_pool, tmp_object_filter);
                end
                 object_num = size(object_filter_pool,3);
                if object_num < Tn  %% update the cur filter with a learning rate before frame Tn;
                    object_filter = (1 - learning_rate) * object_filter + learning_rate*tmp_object_filter;
                    if object_num/frame < 5*Tu %% whether the CUR filter is effective in these scenes;
                        use_object_filter = false;  
                    end
                else   
                    %%learn the CUR filter through random sampling;
                    tmp_order = 1:object_num;
                    pool_order = randperm(numel(tmp_order));
                    col_matrix = object_filter_pool(:,:,pool_order(1:K));
                    object_filter = sum(col_matrix,3)./object_num;
                end
            end
         end
        
        if use_scale_estimate
        % calculate the new target size
            target_sz = floor(base_target_sz * currentScaleFactor);
        end
        %save position and timing
		positions(frame,:) = pos;
		time = time + toc();
        %output_positions(frame,:) = [round(pos([2,1]) - target_sz([2,1])/2+0.0001), round(pos([2,1]) +target_sz([2,1])/2)];  %% princenton dataset
       output_positions(frame,:) = [round(pos([2,1]) - target_sz([2,1])/2+0.0001), target_sz([2,1])];
        if resize_image,
            %output_positions(frame,:) = [round(pos([2,1])*2 - target_sz([2,1])), round(pos([2,1])*2 + target_sz([2,1]))];  %% princenton dataset
            output_positions(frame,:) = [round(pos([2,1])*2 - target_sz([2,1])),target_sz([2,1])*2];
        end
        
        %visualization
        if show_visualization,
            box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            if frame == 1,  %first frame, create GUI
                figure('Number','off', 'Name',['Tracker - ' video_path]);
                im_handle = imshow(uint8(img), 'Border','tight', 'InitialMag', 100 + 100 * (length(img) < 500));
                rect_handle = rectangle('Position',box, 'EdgeColor','g');
                text_handle = text(10, 10, int2str(frame));
                set(text_handle, 'color', [0 1 1]);
            else
                try  %subsequent frames, update GUI
                    set(im_handle, 'CData', img)
                    set(rect_handle, 'Position', box)
                    set(text_handle, 'string', int2str(frame));
                catch
                    return
                end
            end
            drawnow
        end
    end
	if resize_image,        
		positions = positions * 2;
	end
end

