function new_samples = mine_image(image_id, dataset, ...
	class, weights, bias, object_sz, patch_sz, cell_size, ...
	features, detection, mining_threshold, padding)
%MINE_IMAGE
%   Search a single image for hard negatives. The detector is applied to
%   the image, and features for new samples are extracted at each
%   detection.
%
%   Joao F. Henriques, 2013

	%load image to run detector on
	[gt_rects, im] = dataset_image(dataset, class, image_id);

	%skip image if it contains some boxes of this class.
	if ~isempty(gt_rects),
		new_samples = [];
		return
	end

	%run detector
	boxes = detect(im, weights, bias, object_sz, cell_size, features, detection, mining_threshold);

	%to store mined samples
	new_samples = zeros(numel(weights), size(boxes,1));
	is_valid = true(size(boxes,1),1);  %whether each sample is valid

	%process any false positives
	for p = 1:size(boxes,1),
		%x and y coordinates to extract, with padding in all directions
		x1 = floor(boxes(p,1) - padding(2)/2 * boxes(p,3));
		x2 = floor(boxes(p,1) + (1 + padding(2)/2) * boxes(p,3));
		xs = x1 : x2;

		y1 = floor(boxes(p,2) - padding(1)/2*boxes(p,4));
		y2 = floor(boxes(p,2) + (1 + padding(1)/2) * boxes(p,4));
		ys = y1 : y2;

	  	%skip this sample if it includes pixels outside the image
		if x1 < 1 || y1 < 1 || x2 > size(im,2) || y2 > size(im,1),
			is_valid(p) = false;
			continue
		end

		patch = im(ys, xs, :);  %extract the patch

		%resize to the common size (assumes aspect ratio is similar)
		patch = imresize(patch, patch_sz, 'bilinear');

		%extract features (e.g., HOG)
		sample = get_features(patch, features, cell_size);

		%store it
		new_samples(:,p) = sample(:);
	end

	%store valid samples
	new_samples = new_samples(:,is_valid);
end

