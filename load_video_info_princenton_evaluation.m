function [img_files, pos, target_sz, video_path] = load_video_info_princenton_evaluation(base_path, videoname)
video_path = [base_path videoname '/'];
framesInfo = load([video_path, 'frames.mat']);
basicInfo = framesInfo.frames;

frames = zeros(2,1);
frames(1) = 1; frames(2) = basicInfo.length;

%text_files = dir([video_path '*_gt.txt']);
text_files = dir([video_path 'init.txt']);
assert(~isempty(text_files), 'No initial position and ground truth to load.')

f = fopen([video_path text_files(1).name]);
ground_truth = textscan(f, '%f,%f,%f,%f');  %[x, y, width, height]
%the format is [x, y, width, height]
% try
% 	ground_truth = textscan(f, '%f,%f,%f,%f,%f', 'ReturnOnError',false);  
% catch  %%ok, try different format (no commas)
% 	frewind(f);
% 	ground_truth = textscan(f, '%f,%f,%f,%f,%f');  
% end
ground_truth = cat(2, ground_truth{:});
fclose(f);

%set initial position and size
target_sz = [ground_truth(1,4), ground_truth(1,3)];
%pos = [ground_truth(1,2), ground_truth(1,1)];
pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
if size(ground_truth,1) == 1,
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
else
		%store positions instead of boxes
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
end
%ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];

%see if they are in the 'imgs' subfolder or not
if exist(fullfile(video_path, sprintf('rgb/r-%d-%d.png', basicInfo.imageTimestamp(1), basicInfo.imageFrameID(1))))
    imgFiles = cell(frames(2),1);
    for frameId = frames(1):frames(2);
        imgFiles{frameId} = sprintf('rgb/r-%d-%d.png', basicInfo.imageTimestamp(frameId), basicInfo.imageFrameID(frameId));
    end

else
    error('No image files to load.')
end

%list the files
img_files = cellstr(imgFiles);

end

