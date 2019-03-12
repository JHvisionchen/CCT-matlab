function progress(varargin)
%PROGRESS
%   Prints a progress bar on the command window. The look is intentionally
%   simple (a few dots), which makes it compatible with PARFOR without any
%   synchronization code (the order of dot updates does not matter).
%
%   PROGRESS <optional text>
%     Initializes the progress bar with some optional text.
%
%   PROGRESS(ITERATION, TOTAL)
%     Updates the progress bar for ITERATION out of TOTAL iterations.
%
%   Example:
%     progress Running
%     parfor i = 1:20  % or a simple FOR
%       progress(i, 20)
%       pause(0.1)
%     end
%
%   Joao F. Henriques, 2013

	%initialization, print initial line break, and optional strings
	if nargin == 0,
		fprintf('\n')
	elseif ischar(varargin{1}),
		for i = 1:nargin, fprintf([varargin{i} ' ']), end
		fprintf('\n')
	else
		%print 10 dots, evenly spaced across iterations.
		%order does not matter; the number of dots always increases.
		if mod(varargin{1} - 1, ceil(varargin{2} / 10)) == 0,
			%erase previous line break, print dot and next line break.
			%this is necessary because text output from different worker
			%threads can be separated by unpredictable line breaks.
			fprintf('\b.\n')
		end
	end
end

