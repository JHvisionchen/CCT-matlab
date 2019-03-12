%%demo
addpath(genpath('.')); 
%%fix the random generator
stream = RandStream('mt19937ar', 'seed',5489);
RandStream.setGlobalStream(stream);

scale_step = 1.02; scale_range = 33;
params.scale_step = scale_step; params.scale_range = scale_range;
run_tracker('choose','gaussian', 'hog_scale', params); 

