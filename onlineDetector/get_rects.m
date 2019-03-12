function rects = get_rects(response, max_peaks, threshold, ...
	detection_size, suppressed_scale)
%GET_RECTS
%   RECTS = GET_RECTS(RESPONSE, MAX_PEAKS, THRESHOLD, DETECTION_SIZE,
%     SUPPRESSED_SCALE)
%   Iteratively obtain the strongest response peaks, while performing
%   non-maximum suppression (to prevent duplicated detections around a
%   small area). Returns rectangles in the form [x, y, width, height,
%   score].
%
%   Joao F. Henriques, 2013

	%how many cells are suppressed on each side (a vector of 2 elements,
	%both for vertical and horizontal)
	suppressed = max(1, ceil(detection_size * suppressed_scale / 2 - 1));

	%returned rectangles: [x, y, w, h, score]
	rects = zeros(max_peaks, 5);

	for k = 1:max_peaks,
		%find peak and get its coordinates
		[peak_value, index] = max(response(:));
		[y,x] = ind2sub(size(response), index);
		
		if peak_value < threshold || ~isfinite(peak_value),  %early exit
			rects = rects(1:k-1,:);
			break;
		end
		
		%clear a rectangle around the peak
		x1 = max(x - suppressed(2), 1);
		x2 = min(x + suppressed(2), size(response,2));
		y1 = max(y - suppressed(1), 1);
		y2 = min(y + suppressed(1), size(response,1));
		response(y1:y2,x1:x2) = -inf;
		
		%store rectangle
		rects(k,:) = [x + 0.5 - detection_size(2) / 2, ...
			y + 0.5 - detection_size(1) / 2, ...
			detection_size(2), detection_size(1), peak_value];
	end

end
