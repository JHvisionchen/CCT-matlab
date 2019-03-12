function discriminant_detector = train_discriminant_detector(samples, num_pos_samples)

sz = size(samples);
N = sqrt(prod(sz(1:2)));  %constant factor that makes the FFT/IFFT unitary

target_magnitude = 1;
%profile of labels for negative samples, for each shift
neg_labels = -target_magnitude * ones(sz(1:2));  %same label for all samples

%profile of labels for positive samples, for each shift
target_sigma = 0.5;
pos_labels = gaussian_shaped_labels_detection(target_magnitude, target_sigma, sz);

pos_labels = fft2(pos_labels) / N;
neg_labels = fft2(neg_labels) / N;
samples = fft2(samples) / N;

%set constant frequency of labels to 0 (equivalent to subtracting the mean)
pos_labels(1,1) = 0;
neg_labels(1,1) = 0;

weights = zeros(sz(1:3));
bias = 0;

% %enable parallel execution
% if parallel && matlabpool('size') == 0,
% 	matlabpool open
% end
%train a complex SVR (liblinear) with no bias term
training.type = 'svr';
training.regularization = 1e-2;  %SVR-C, obtained by cross-validation on a log. scale
training.epsilon = 1e-2;
training.complex = true;
training.bias_term = 0;
%if ~parallel,
if 1
	%circulant decomposition (non-parallel code).
	
	y = zeros(sz(4),1);  %sample labels (for a fixed frequency)
	progress Training
% 	tic
	for r = 1:sz(1),
		for c = 1:sz(2),
			%fill vector of sample labels for this frequency
			y(:) = neg_labels(r,c);
			y(1:num_pos_samples) = pos_labels(r,c);

			%train classifier for this frequency
			weights(r,c,:) = linear_training(training, double(permute(samples(r,c,:,:), [4, 3, 1, 2])), y);
		end
		
		progress(r, sz(1));
	end
% 	toc
	
else
	%circulant decomposition (parallel code).
	%to use "parfor", we have to deal with a number of issues.
	
	%first, split data into chunks, stored in a cell array, to avoid the
	%2GB data-transfer limit of "parfor" (bug fixed in MATLAB2013a). (*)
% 	disp('Chunking data to avoid MATLAB''s data transfer limit...')
	samples_chunks = cell(sz(1),1);
	for r = 1:sz(1),
		samples_chunks{r} = samples(r,:,:,:);
	end
	clear samples;
	
	progress Training
% 	tic
	parfor r = 1:sz(1),
		%normally we'd set "weights(r,c,:)" as in the non-parallel code,
		%but "parfor" doesn't like complicated indexing. so the inner loop
		%will build just one row, and only then we store it in "weights".
		row_weights = zeros(sz(2:3)); 

		y = zeros(sz(4),1);  %sample labels (for a fixed frequency)

		for c = 1:sz(2),
			%fill vector of sample labels for this frequency
			y(:) = neg_labels(r,c);
			y(1:num_pos_samples) = pos_labels(r,c);

			row_weights(c,:) = linear_training(training, double(permute(samples_chunks{r}(1,c,:,:), [4, 3, 1, 2])), y);

% 			%with MATLAB2013a or newer, you can comment out the chunking
% 			%code (*), and use this to train with "samples" directly:
% 			row_weights(c,:) = linear_training(training, double(permute(samples(r,c,:,:), [4, 3, 1, 2])), y);
		end

		weights(r,:,:) = row_weights;  %store results for this row of weights

		progress(r, sz(1));
	end
% 	toc
end

%transform solution back to the spatial domain
weights = real(ifft2(weights)) * N;

discriminant_detector = weights;
% if 1,
% 	%weights visualization (positive on the left, negative on the right)
% 	w_norm = max(abs(weights(:)));
% 	figure('Name', ['Weight range: ' num2str(w_norm) ', bias: ' num2str(bias)])
% 	try  %#ok<ALIGN>
% 		imshow(vl_hog('render', 0.4 * single([max(0, weights / w_norm), max(0, -weights / w_norm)])));
% 	catch, end  %#ok<CTCH>
% end