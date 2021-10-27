function runPredictImg(Img)
% Basic function to test out running randomized image prediction algorithm and gives us predictive values

X = imread(Img); % reads the image .bmp (24 bits) (20x20)
X = double(X); % converts it to double
temp = X; % creates a copy for later use
X = (X.-128)./255; % normalize the features
X = X .* (temp > 0); % return the original 0 values to the X
X = reshape(X, [], numel(X)); % converts the 20x20 matrix into a 1x400 vector

val = randi(50);
Theta1 = rand(val, length(X)+1);
Theta2 = rand(randi(10), val+1);
predictImg(Theta1, Theta2, Img) 

end