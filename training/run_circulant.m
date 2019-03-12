
% This is the main script for training with the Circulant Decomposition.
% It has both single-threaded and parallel implementations.
% It assumes all parameters have been set (see run_circulant_*.m files).
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



sz = size(samples);
N = sqrt(prod(sz(1:2)));  %constant factor that makes the FFT/IFFT unitary

%profile of labels for negative samples, for each shift
neg_labels = -target_magnitude * ones(sz(1:2));  %same label for all samples

%profile of labels for positive samples, for each shift
pos_labels = gaussian_shaped_labels(target_magnitude, target_sigma, sz);




% Data dimensions:
%          size(samples) = [height, width, hog bins, samples]
% "fft2" will Fourier-transform the first 2 dimensions only. Instead of
% spatial cells, they will then correspond to Fourier frequencies. For
% each frequency "samples(r,c,:,:)", we get a slice with dimensions
% [1, 1, bins, samples]. The function "permute" makes it [samples, bins].
% This is now a valid data matrix that can be used to train an SVM/SVR.


%transform all data (including training labels) to the Fourier domain
disp('Running FFT...')
tic
pos_labels = fft2(pos_labels) / N;
neg_labels = fft2(neg_labels) / N;
samples = fft2(samples) / N;
toc

%set constant frequency of labels to 0 (equivalent to subtracting the mean)
pos_labels(1,1) = 0;
neg_labels(1,1) = 0;


weights = zeros(sz(1:3));
bias = 0;

if ~parallel,
	%circulant decomposition (non-parallel code).
	
	y = zeros(sz(4),1);  %sample labels (for a fixed frequency)
	progress Training
	tic
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
	toc
	
else
	%circulant decomposition (parallel code).
	%to use "parfor", we have to deal with a number of issues.
	
	%first, split data into chunks, stored in a cell array, to avoid the
	%2GB data-transfer limit of "parfor" (bug fixed in MATLAB2013a). (*)
	disp('Chunking data to avoid MATLAB''s data transfer limit...')
	samples_chunks = cell(sz(1),1);
	for r = 1:sz(1),
		samples_chunks{r} = samples(r,:,:,:);
	end
	clear samples;
	
	progress Training
	tic
	parfor r = 1:sz(1),
		%normally we'd set "weights(r,c,:)" as in the non-parallel code,
		%but "parfor" doesn't like complicated indexing. so the inner loop
		%will build just one row, and only then we store it in "weights".
		row_weights = zeros(sz(2:3));  %#ok<PFBNS>

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
	toc
end

%transform solution back to the spatial domain
weights = real(ifft2(weights)) * N;


%crop template by some cells if needed
crop = floor(cropping_cells / 2);
weights = weights(1 + crop : end - crop, 1 + crop : end - crop, :);
assert(~isempty(weights), 'Too much cropping.')

if show_plots,
	%weights visualization (positive on the left, negative on the right)
	w_norm = max(abs(weights(:)));
	figure('Name', ['Weight range: ' num2str(w_norm) ', bias: ' num2str(bias)])
	try  %#ok<ALIGN>
		imshow(vl_hog('render', 0.4 * single([max(0, weights / w_norm), max(0, -weights / w_norm)])));
	catch, end  %#ok<CTCH>
end

save_file_name = ['circulant_' dataset_name '_' class];

if save_weights,  %save template weights to a MAT file
	if ~exist([paths.cache 'weights/'], 'dir'),
		mkdir([paths.cache 'weights/'])
	end
	save([paths.cache 'weights/' save_file_name '_weights'], 'weights', 'bias', 'object_sz')
end

%run evaluation
evaluate_detector(dataset, class, weights, bias, object_sz, cell_size, features, ...
	detection, paths, save_file_name, save_plots, show_plots, show_detections, parallel);

