function [img_files, pos, target_sz, ground_truth, video_path] = load_video_info_color(base_path, video)

% [img_files, pos, target_sz, ground_truth, video_path] = load_video_info(video_path)
video_path = [base_path video '/'];
text_files = dir([video_path '*_frames.txt']);
f = fopen([video_path text_files(1).name]);
frames = textscan(f, '%f,%f');
fclose(f);

%text_files = dir([video_path '*_gt.txt']);
text_files = dir([video_path 'groundtruth_rect.txt']);
assert(~isempty(text_files), 'No initial position and ground truth to load.')

f = fopen([video_path text_files(1).name]);
%ground_truth = textscan(f, '%f,%f,%f,%f');  %[x, y, width, height]
%the format is [x, y, width, height]
try
	ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
catch  %#ok, try different format (no commas)
	frewind(f);
	ground_truth = textscan(f, '%f %f %f %f');  
end
ground_truth = cat(2, ground_truth{:});
fclose(f);

%set initial position and size
target_sz = [ground_truth(1,4), ground_truth(1,3)];
%pos = [ground_truth(1,2), ground_truth(1,1)];
pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);

% ground_truth = [ground_truth(:,[2,1]) + (ground_truth(:,[4,3]) - 1) / 2 , ground_truth(:,[4,3])];

%see if they are in the 'imgs' subfolder or not
if exist([video_path num2str(frames{1}, 'img/%04i.png')], 'file'),
    video_path = [video_path 'img/'];
    img_files = num2str((frames{1} : frames{2})', '%04i.png');
elseif exist([video_path num2str(frames{1}, 'img/%04i.jpg')], 'file'),
    video_path = [video_path 'img/'];
    img_files = num2str((frames{1} : frames{2})', '%04i.jpg');
elseif exist([video_path num2str(frames{1}, 'imgs/img%04i.bmp')], 'file'),
    video_path = [video_path 'img/'];
    img_files = num2str((frames{1} : frames{2})', '%04i.bmp');
else
    error('No image files to load.')
end

%list the files
img_files = cellstr(img_files);

end

