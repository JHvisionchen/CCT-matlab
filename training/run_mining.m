
% This is the main script for training with Hard Negative Mining.
% It has both single-threaded and parallel implementations.
% It assumes all parameters have been set (see run_mining_*.m files).
%
% Joao F. Henriques, 2013


%initialize dataset information
dataset = dataset_init(dataset_name, paths);


load_samples;


%enable parallel execution
if parallel && matlabpool('size') == 0,
	matlabpool open
end
%make sure the needed functions are in the path of all workers (work around a MATLAB bug)
addpath('./detection', './evaluation', './training', './utilities', './libraries')


disp('Initial training...')

% Data dimensions:
%          size(samples) = [height, width, hog bins, samples]
% Reshape it into a 2D data matrix:
%          size(X) = [height * width * hog bins, samples]

sz = size(samples);
X = double(reshape(samples, prod(sz(1:3)), sz(4)).');

%vector of sample labels
y = ones(sz(4),1);
y(num_pos_samples+1:end) = -1;

%learn initial template
tic
[weights, bias] = linear_training(training, X, y);
toc
weights = reshape(weights, sz(1:3));  %reshape from vector to an actual template

%negative images for mining
images = dataset_list(dataset, 'train', class, false);

%do some rounds of hard negative mining
for current_round = 1:hard_neg_rounds,
	
	%if necessary, evaluate the previously trained detector at each round
	if eval_all_rounds,
		save_file_name = ['mining_' dataset_name '_' class '_' int2str(current_round - 1)];
		
		evaluate_detector(dataset, class, weights, bias, object_sz, cell_size, ...
			features, detection, paths, save_file_name, save_plots, show_plots, false, parallel);
		
		if show_plots, pause(0.1); end  %to let the previous plot show (more reliable than drawnow)
	end
	
	
	disp(['Hard negative mining round ' int2str(current_round) '...'])
	tic

	%store results in a cell array because we don't know the size a priori
	mined = cell(numel(images),1);
	
	progress();  %initialize progress bar

	if ~parallel,
		%search each image for hard negatives
		for f = 1:numel(images),
			mined{f} = mine_image(images{f}, dataset, class, weights, ...
				bias, object_sz, patch_sz, cell_size, features, ...
				detection, mining_threshold, padding);

			progress(f, numel(images))
		end
		
	else
		%exactly the same, but using "parfor"
		parfor f = 1:numel(images),
			mined{f} = mine_image(images{f}, dataset, class, weights, ...
				bias, object_sz, patch_sz, cell_size, features, ...
				detection, mining_threshold, padding);

			progress(f, numel(images))  %#ok<PFBNS>
		end
	end
	
	%flatten the cell array of samples into a matrix
	mined = cat(2, mined{:}).';

	%append to current data matrix
	X = [X; double(mined)];  %#ok
	y = [y; -ones(size(mined,1), 1)];  %#ok

	%re-train with new set
	[weights, bias] = linear_training(training, X, y);
	weights = reshape(weights, sz(1:3));

	toc
end


if show_plots,
	%weights visualization (positive on the left, negative on the right)
	w_norm = max(abs(weights(:)));
	figure('Name', ['Weight range: ' num2str(w_norm) ', bias: ' num2str(bias)])
	try  %#ok<ALIGN>
		imshow(vl_hog('render', 0.4 * single([max(0, weights / w_norm), max(0, -weights / w_norm)])));
	catch, end  %#ok<CTCH>
end

save_file_name = ['mining_' dataset_name '_' class '_' int2str(hard_neg_rounds)];

if save_weights,  %save template weights to a MAT file
	if ~exist([paths.cache 'weights/'], 'dir'),
		mkdir([paths.cache 'weights/'])
	end
	save([paths.cache 'weights/' save_file_name '_weights'], 'weights', 'bias', 'object_sz')
end

%run evaluation on the final detector
evaluate_detector(dataset, class, weights, bias, object_sz, cell_size, features, ...
	detection, paths, save_file_name, save_plots, show_plots, show_detections, parallel);
