function overlaps = compute_overlap(A, B)
%calculate the overlap in each dimension
overlap_height = min(A(:,1) + A(:,3)/2, B(:,1) + B(:,3)/2) ...
    - max(A(:,1) - A(:,3)/2, B(:,1) - B(:,3)/2);
overlap_width = min(A(:,2) + A(:,4)/2, B(:,2) + B(:,4)/2) ...
    - max(A(:,2) - A(:,4)/2, B(:,2) - B(:,4)/2);

% if no overlap, set to zero
overlap_height(overlap_height < 0) = 0;
overlap_width(overlap_width < 0) = 0;

% remove NaN values (should not exist any)
valid_ind = ~isnan(overlap_height) & ~isnan(overlap_width);

% calculate area
overlap_area = overlap_height(valid_ind) .* overlap_width(valid_ind);
tracked_area = A(3) .* A(4);
ground_truth_area = B(valid_ind,3) .* B(valid_ind,4);

% calculate PASCAL overlaps
overlaps = overlap_area ./ (tracked_area + ground_truth_area - overlap_area);