function plotEx1Data1()
%PLOTEX1DATA1 Plots the data points x and y into a new figure 
%   PLOTEX1DATA1() plots the data points and gives the figure axes labels

figure('name', 'ex1data1 Data Plots'); % open a new figure window

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

% Constants
ROWS=1;
COLS=3;

% Plot ex1data1.txt
load ex1data1.txt;
x = ex1data1(:,1);
y = ex1data1(:,2);

% Plot gradient descent fit of ex1data1.txt
subplot(ROWS,COLS,1);
hold on;
plot(x, y, 'rx', 'MarkerSize', 5);
[theta J_hist] = gradientDescent(x, y, [0], 0.001, 1000);
y_guess = theta' * x;
plot(x, y_guess)
xlabel("Input values");
ylabel("Output values");
title("Gradient Descent fit");

% Plot regression iteration costs
subplot(ROWS,COLS,2);
x_iter = (1:1000)';
plot(x_iter, J_hist);
xlabel("Number of iterations");
ylabel("Cost");
title("Gradient Descent Cost per Iteration");

% Plot Normal equation fit
subplot(ROWS,COLS,3);
hold on;
plot(x, y, 'rx', 'MarkerSize', 5);
theta = normalEqn(x,y);
y_normal = theta' * x;
plot(x, y_normal);
xlabel("Input values");
ylabel("Output values");
title("Normal equation fit");

% ============================================================

end
