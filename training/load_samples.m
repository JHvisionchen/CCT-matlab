
% This script loads all samples (positive and negative) for the current
% object CLASS in the dataset (initialized with DATASET_INIT). It is
% meant to be called by either RUN_MINING or RUN_CIRCULANT.
%
% To avoid processing the images every time, the samples are cached in
% MAT files (you can set the path in SET_DEFAULTS). The cache will be
% refreshed automatically if any relevant parameters change.
%
% Joao F. Henriques, 2013


clear samples;  %avoid stressing memory use


%cache samples in a MAT file, to avoid processing the images every time
if circulant,
	cache_file = [paths.cache 'circulant_samples_' dataset_name '_' class '.mat'];
else
	cache_file = [paths.cache 'mining_samples_' dataset_name '_' class '.mat'];
end

%current parameters, to compare against the cache file
new_parameters = struct('sampling',sampling, 'features',features, 'cell_size',cell_size, ...
	'padding_cells',padding_cells, 'object_size',object_size, 'circulant',circulant);

if exist(cache_file, 'file'),
	load(cache_file)  %load data and parameters
	
	%if the parameters are the same, we're done
	if isequal(parameters, new_parameters),
		disp('Reloaded samples from cache.')
		
		%compute padding, relative to the object size
		padding = (patch_sz - object_sz) ./ object_sz;
		
		return
	end
end


%otherwise, start from scratch. first, load the positive samples
load_pos_samples;

num_pos_samples = size(pos_samples,4);
sample_sz = size(pos_samples);

%now the negatives
disp('Loading negative samples...')

%list training image files for all classes
images = dataset_list(dataset, 'train', class, false);  %skip images with this class
n = numel(images);

if circulant,
	%dense sampling

	%stride size, in cells (vertical and horizontal directions)
	stride_sz = floor(sampling.neg_stride * sample_sz(1:2));

	%compute max. number of samples given the image size and stride
	%(NOTE: this is probably a pessimistic estimate!)
	num_neg_samples = numel(images) * prod(floor(sampling.neg_image_size / cell_size ./ stride_sz));

	%initialize data structure for all samples, starting with positives
	samples = cat(4, pos_samples, zeros([sample_sz(1:3), num_neg_samples], 'single'));

	progress();

	idx = num_pos_samples + 1;  %index of next sample

	for f = 1:n,
		%load image and bounding box info
		[boxes, im] = dataset_image(dataset, class, images{f});

		%ensure maximum size
		if size(im,1) > sampling.neg_image_size(1),
			im = imresize(im, [sampling.neg_image_size(1), NaN], 'bilinear');
		end
		if size(im,2) > sampling.neg_image_size(2),
			im = imresize(im, [NaN, sampling.neg_image_size(2)], 'bilinear');
		end

		%extract features (e.g., HOG)
		x = get_features(im, features, cell_size);

		%extract subwindows, given the specified stride
		for r = 1 : stride_sz(1) : size(x,1) - sample_sz(1) + 1,
			for c = 1 : stride_sz(2) : size(x,2) - sample_sz(2) + 1,
				%store the sample
				samples(:,:,:,idx) = x(r : r+sample_sz(1)-1, c : c+sample_sz(2)-1, :);
				idx = idx + 1;
			end
		end

		progress(f, n);
	end

else
	%random sampling

	%use a fixed seed so we should see the exact same results every
	%time. we use the old syntax so results are the same in most PCs,
	%but we may have to fallback on the new syntax in the future.
	try
		rand('seed', 0);  %#ok<RAND>
	catch  %#ok<CTCH>
		rng(0);  %new syntax
	end

	num_neg_samples = sampling.neg_samples_per_image * n;

	%initialize data structure for all samples, starting with positives
	samples = cat(4, pos_samples, zeros([sample_sz(1:3), num_neg_samples], 'single'));

	progress();

	idx = num_pos_samples + 1;  %index of next sample

	for f = 1:n,
		%load image and bounding box info
		[boxes, im] = dataset_image(dataset, class, images{f});

		%extract features (e.g., HOG)
		x = get_features(im, features, cell_size);

		%if image isn't big enough for a full patch, skip it
		if size(x,1) < sample_sz(1) || size(x,2) < sample_sz(2),
			continue
		end

		for s = 1:sampling.neg_samples_per_image,
			%random position, fully inside image
			r = randi([1, size(x,1) - sample_sz(1)]);
			c = randi([1, size(x,2) - sample_sz(2)]);

			%extract sample and store it
			samples(:,:,:, idx) = x(r : r + sample_sz(1) - 1, c : c + sample_sz(2) - 1, :);
			idx = idx + 1;
		end

		progress(f, n);
	end
end

assert(idx > num_pos_samples + 1, 'No valid negative samples to load.')

%trim any uninitialized samples at the end
if idx - 1 < size(samples,4),
	samples(:,:,:, idx : end) = [];
end

%save the results
if ~exist(paths.cache, 'dir'),
	mkdir(paths.cache)
end
parameters = new_parameters;
try  %use 7.3 format, otherwise Matlab *won't* save matrices >2GB, silently
	save(cache_file, 'samples', 'parameters', 'num_pos_samples', 'patch_sz', 'object_sz', '-v7.3')
catch  %#ok<CTCH>  if it's not supported just use whatever is the default
	save(cache_file, 'samples', 'parameters', 'num_pos_samples', 'patch_sz', 'object_sz')
end
