function [ale, precision, overlap, fps] = run_tracker(video, kernel_type, feature_type, params, show_visualization, show_plots)

	%path to the videos (you'll be able to choose one with the GUI).
    base_path = '/media/cjh/datasets/tracking/OTB100/';
	%default settings
	if nargin < 1, video = 'choose'; end
	if nargin < 2, kernel_type = 'gaussian'; end
	if nargin < 3, feature_type = 'hog'; end
    if nargin < 5, show_visualization = ~strcmp(video, 'all'); end
	if nargin < 6, show_plots = ~strcmp(video, 'all'); end

	%parameters according to the paper. at this point we can override
	%parameters based on the chosen kernel or feature type
	kernel.type = kernel_type;
	
	features.gray = false;
	features.hog = false;
    features.part_hog_type = 13;  %% for different hog type
    features.use_rgb2gray = false; 
    features.use_scale_estimate = true;  %%whether using the scale estimate
    features.use_object_detector = true; %% whether using the CUR filter
	
	padding = 1.5;  %extra area surrounding the target
	lambda = 1e-4;  %regularization
	output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

	switch feature_type
	case 'gray',
		interp_factor = 0.075;  %linear interpolation factor for adaptation

		kernel.sigma = 0.2;  %gaussian kernel bandwidth
		
		kernel.poly_a = 1;  %polynomial kernel additive term
		kernel.poly_b = 7;  %polynomial kernel exponent
	
		features.gray = true;
		cell_size = 1;
		
	case 'hog',
		interp_factor = 0.02;
		
		kernel.sigma = 0.5;
		
		kernel.poly_a = 1;
		kernel.poly_b = 9;
		
		features.hog = true;
		features.hog_orientations = 9;
		cell_size = 4;
        
        features.use_rgb2gray  = true;  %%original version
        
   case 'hog_scale',
		interp_factor = 0.02;
		
		kernel.sigma = 0.5;
		
		kernel.poly_a = 1;
		kernel.poly_b = 9;
		
		features.hog = true;
		features.hog_orientations = 9;
		cell_size = 4;
        
        features.use_rgb2gray  = true;  %%original version
        features.use_scale_estimate = true;
	
	otherwise
		error('Unknown feature.')
    end
	assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')
    
	switch video
	case 'choose',
		%ask the user for the video, then call self with that video name.
		video = choose_video(base_path);
		if ~isempty(video),
			[ale, precision, overlap, fps] = run_tracker(video, kernel_type, feature_type,params, show_visualization, show_plots);
			
			if nargout == 0,  %don't output precision as an argument
				clear precision
			end
        end		
		
	case 'benchmark',
		seq = evalin('base', 'subS');
		target_sz = seq.init_rect(1,[4,3]);
		pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);
		img_files = seq.s_frames;
		video_path = [];
		
		%call tracker function with all the relevant parameters
		positions = tracker(video_path, img_files, pos, target_sz, padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, features, false);		
		%return results to benchmark, in a workspace variable
		rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
		rects(:,3) = target_sz(2);
		rects(:,4) = target_sz(1);
		res.type = 'rect';
		res.res = rects;
		assignin('base', 'res', res);		
		
	otherwise
		%we were given the name of a single video to process.
	
		%get image file names, initial state, and ground truth for evaluation
		[img_files, pos, target_sz, ~, video_path] = load_video_info(base_path, video);
		
		%call tracker function with all the relevant parameters
		[~, ~, time] = tracker(video_path, img_files, pos, target_sz, padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, features,params, show_visualization);

        fps = numel(img_files) / time;

		if nargout > 0,
            precision = 0;
            overlap = 0;
            ale = 0;          
        end
	end
end
