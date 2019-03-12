function visualize_detections(dataset, class, images, detections, labels, threshold)
%VISUALIZE_DETECTIONS
%   Shows an interactive figure visualization of detection results.
%
%   The scrollbar at the bottom allows navigation between images. For a
%   list of keyboard shortcuts, see HELP VIDEOFIG.
%
%   VISUALIZE_DETECTIONS(DATASET, CLASS, IMAGES, DETECTIONS, LABELS,
%     THRESHOLD)
%   Displays detection results of DETECTIONS cell array, one cell per
%   image, with N-by-5 arrays of bounding box information. Each row is in
%   the format [x, y, width, height, score]. The results are for object
%   CLASS of the dataset specified in the DATASET struct (returned by
%   "dataset_init"). IMAGES is the cell array that identifies all test set
%   images (returned by "dataset_list"). LABELS is a cell array,
%   containing one vector of labels for each image, indicating whether each
%   detection is a true or false positive (computed with "evaluate_image").
%   Only detections with scores above THRESHOLD will be displayed.
%
%   Joao F. Henriques, 2013

	n = numel(images);  %number of frames

	%remember lowest and highest score, to map them to the range [0, 1].
	%this way we can show a detection's score as its bounding box opacity.
	high_score = -inf(n,1);  %highest in each frame
	low_score = inf(n,1);  %lowest in each frame
	for k = 1:n,
		if ~isempty(detections{k}),
			high_score(k) = max(detections{k}(:,5));
			low_score(k) = min(detections{k}(:,5));
		end
	end
	%record highest/lowest across frames. also, start the interface at the
	%frame where we have the best scoring detection.
	[high_score, initial_frame] = max(high_score);
	low_score = max(low_score);
	

	%create video interface
	im_h = [];  %image handle
	[fig_h, axes_h, scrollbar_h, scroll] = videofig(n, @redraw);
	set(fig_h, 'Number','off', 'Name',['Detections for ' class ' in ' dataset.name])
	
	%show a helpful text tip on the scrollbar
	text_h = text(0.5, 0.5, 'Click and drag to navigate images.', 'Color','w', ...
		'HitTest','off', 'Horiz','center', 'Vert','middle', 'Parent',scrollbar_h(1));
	
	%go to initial frame and draw it
	scroll(initial_frame);
	
	function redraw(f)
		[gt_boxes, im] = dataset_image(dataset, class, images{f});
		
		%resize image to a decent size and remember scale, to rescale boxes
		old_height = size(im,1);
		if size(im,2) > size(im,1),
			im = imresize(im, [NaN, 640], 'bilinear');
		else
			im = imresize(im, [480, NaN], 'bilinear');
		end
		scale = size(im,1) / old_height;
		
		%draw ground truth in black
		for i = 1:size(gt_boxes,1),
			im = draw_box(im, gt_boxes(i,:) * scale, [0, 0, 0], 0.5, 2);
		end
		
		for i = 1:size(detections{f},1),
			box = detections{f}(i,:);
			
			%opacity will be 0 for the lowest score (or threshold if
			%specified), and 1 for the highest score.
			opacity = (box(5) - max(low_score, threshold)) / (high_score - threshold);
			
			box = box(1:4) * scale;  %keep only the coordinates and scale
			
			if opacity > 0,  %don't draw negative values
				if labels{f}(i) > 0,  %true positive, green with thin black outline
					im = draw_box(im, box + [-1, -1, 2, 2], [0, 0, 0], opacity, 1);
					im = draw_box(im, box + [2, 2, -4, -4], [0, 0, 0], opacity, 1);

					im = draw_box(im, box, [64, 255, 64], opacity, 2);

				else  %false positive, red
					im = draw_box(im, box, [192, 0, 0], opacity, 2);
				end
			end
		end
		
		
		if isempty(im_h) || ~isequal(size(get(im_h, 'CData')), size(im)),
			%first time or different sized image, create image object
			im_h = imshow(im, 'Border','tight', 'Parent',axes_h);
		else
			%if same size just update it
			set(im_h, 'CData', im);
		end
		
		%remove text tip after the first time
		if ~isempty(text_h) && f ~= initial_frame,
			delete(text_h);
			text_h = [];
		end
	end

end

%we render boxes directly in the image pixels, which is much faster than
%creating and deleting several rectangle graphics handles per frame.

function im = draw_box(im, box, color, opacity, line_width)
	%box is [x, y, width, height].
	x1 = floor(box(1));
	y1 = floor(box(2));
	x2 = floor(box(1) + box(3));
	y2 = floor(box(2) + box(4));
	d = line_width - 1;
	
	%horizontal stripes
	im = filled_rectangle(im, x1+1+d, y1, x2-1-d, y1+d, color, opacity);
	im = filled_rectangle(im, x1+1+d, y2-d, x2-1-d, y2, color, opacity);
	
	%vertical stripes
	im = filled_rectangle(im, x1, y1, x1+d, y2, color, opacity);
	im = filled_rectangle(im, x2-d, y1, x2, y2, color, opacity);

end

function im = filled_rectangle(im, x1, y1, x2, y2, color, opacity)
	%get coordinates inside the image
	x1 = max(1, x1);
	y1 = max(1, y1);
	x2 = min(size(im,2), x2);
	y2 = min(size(im,1), y2);
	
	for c = 1:size(im,3),  %iterate R, G and B
		im(y1:y2, x1:x2, c) = (1 - opacity) * im(y1:y2, x1:x2, c) + opacity * color(c);
	end
end

