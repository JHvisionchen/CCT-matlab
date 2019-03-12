function rects = post_process_rects(rects, max_overlap, clip_rect)
%POST_PROCESS_RECTS
%   Delete duplicated detections (non-maximum suppression).
%
%   RECTS = POST_PROCESS_RECTS(RECTS, MAX_OVERLAP)
%   Deletes duplicated detections (rectangles) based on an overlap
%   criterion. When in conflict, only the detections with the strongest
%   responses are kept.
%
%   RECTS = POST_PROCESS_RECTS(RECTS, MAX_OVERLAP, CLIP_RECT)
%   Detections partially outside a given CLIP_RECT will also be deleted.
%
%   Joao F. Henriques, 2013

	if nargin < 3,
		valid = true(size(rects,1), 1);
	else
% 		%only rects that are at least partially inside clip_rect are valid
% 		valid = (rectint(rects(:,1:4), clip_rect) > 0);

		%only rects that are fully inside clip_rect are valid
		valid = (rects(:,1) >= clip_rect(1) & ...
			rects(:,2) >= clip_rect(2) & ...
			rects(:,1) + rects(:,3) <= clip_rect(1) + clip_rect(3) & ...
			rects(:,2) + rects(:,4) <= clip_rect(2) + clip_rect(4));
	end
	
	%sort by detector score
	rects = sortrows(rects, 5);
	
	%calculate intersection area for each pair of rects
	intersections = rectint(rects(:,1:4), rects(:,1:4));
	areas = diag(intersections);
	
	for k = 1:size(rects,1),
		%rect invalid if a rect with a higher score has sufficient overlap
		valid(k) = ~any(intersections(k,k+1:end) > max_overlap * areas(k));
	end
	
	%remove invalid rects
	rects = rects(valid,:);

end

