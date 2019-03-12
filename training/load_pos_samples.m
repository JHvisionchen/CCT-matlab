
% This script will load positive samples of object CLASS in the dataset
% (initialized with DATASET_INIT). It is meant to be called by
% LOAD_SAMPLES.
%
% Joao F. Henriques, 2013


debug_pos_samples = false;
% debug_pos_samples = true;  %enable to show visualization


disp('Loading positive samples...')

%list training image files for this class
images = dataset_list(dataset, 'train', class);
n = numel(images);

%first, count positive samples and figure out average aspect ratio
aspect_ratio_sum = 0;
num_pos_samples = 0;
for k = 1:n,
	%load bounding boxes, and compute running sum of aspect ratios
	boxes = dataset_image(dataset, class, images{k});

	aspect_ratio_sum = aspect_ratio_sum + sum(boxes(:,3) ./ boxes(:,4));
	num_pos_samples = num_pos_samples + size(boxes,1);
end
aspect_ratio = aspect_ratio_sum / num_pos_samples;  %average value

assert(num_pos_samples > 0, 'No valid positive samples to load.')

if ~isscalar(object_size),  %both height and width were specified
	object_sz = object_size;
	aspect_ratio = object_sz(2) / object_sz(1);
else
	%we are only given the size of the largest dimension as a reference
	%(width or height), the other must be deduced from the data.
	if aspect_ratio >= 1,  %width >= height, width is fixed to object_size
		object_sz = [floor(object_size / aspect_ratio), object_size];
	else  %width < height, height is fixed to object_size
		object_sz = [object_size, floor(object_size * aspect_ratio)];
	end
end

%enlarge the patch size with padding
patch_sz = object_sz + padding_cells * cell_size;

%make sure the patch size is a multiple of the cell size
patch_sz = ceil(patch_sz / cell_size) * cell_size;

%length of the diagonal
diagonal = sqrt(sum(object_sz.^2));

%total padding, relative to the object size
padding = (patch_sz - object_sz) ./ object_sz;


%extract features (e.g., HOG) of a dummy sample to figure out the size
sample = get_features(zeros(patch_sz), features, cell_size);

%allocate results array
if ~sampling.flip_positives,
	pos_samples = zeros([size(sample), num_pos_samples], 'single');
else  %allocate twice the samples, for flipped versions
	pos_samples = zeros([size(sample), 2 * num_pos_samples], 'single');
end
idx = 1;

if debug_pos_samples, figure, end

progress();

for k = 1:n,
	%load image and ground truth bounding boxes (x,y,w,h)
	[boxes, im] = dataset_image(dataset, class, images{k});


	for p = 1:size(boxes,1),
		%skip samples that have a very different aspect ratio
		ratio = boxes(p,3) / boxes(p,4) / aspect_ratio;
		if ratio > sampling.reject_aspect_ratio || ratio < 1/sampling.reject_aspect_ratio,		  
			continue
		end

		%center coordinates
		xc = boxes(p,1) + boxes(p,3) / 2;
		yc = boxes(p,2) + boxes(p,4) / 2;

% 		%rescale a box with the correct aspect ratio to have the
% 		%same center and same diagonal length as the ground truth
% 		sz = object_sz / diagonal * sqrt(sum(boxes(p,3:4).^2));

		sz = object_sz / object_sz(1) * boxes(p,4);  %rescale to have same height

		%apply padding in all directions
		sz = (1 + padding) .* sz;


		%x and y coordinates to extract. remember all sizes ("sz") are
		%in Matlab's format (rows, columns)
		xs = floor(xc - sz(2) / 2) : floor(xc + sz(2) / 2);
		ys = floor(yc - sz(1) / 2) : floor(yc + sz(1) / 2);

		%avoid out-of-bounds coordinates (set them to the values at
		%the borders)
		bounded_xs = max(1, min(size(im,2), xs));
		bounded_ys = max(1, min(size(im,1), ys));

		patch = im(bounded_ys, bounded_xs, :);  %extract the patch

		%set out-of-bounds pixels to 0
		patch(ys < 1 | ys > size(im,1), xs < 1 | xs > size(im,2), :) = 0;


		%resize to the common size
		patch = imresize(patch, patch_sz, 'bilinear');

		%extract features (e.g., HOG)
		sample = get_features(patch, features, cell_size);

		%store the sample
		pos_samples(:,:,:,idx) = sample;
		idx = idx + 1;


		if sampling.flip_positives,
			%store a horizontally flipped version too
			sample = get_features(patch(:, end:-1:1, :), features, cell_size);
			pos_samples(:,:,:,idx) = sample;
			idx = idx + 1;
		end


		if debug_pos_samples,  %debug_pos visualization
			imshow(patch, 'InitialMag',300)
			rectangle('Position', [patch_sz([2,1])/2 - object_sz([2,1])/2, object_sz([2,1])], 'EdgeColor','g')
			pause
		end
	end


	progress(k, n);
end

%trim any uninitialized samples at the end
num_rejected = size(pos_samples,4) - idx + 1;
if num_rejected > 0,
	pos_samples(:,:,:, idx : end) = [];
end

%print some debug info
disp(['Loaded ' int2str(size(pos_samples,4)) ' positive samples. Rejected '...
	int2str(num_rejected) ' (wrong aspect ratio).']);

