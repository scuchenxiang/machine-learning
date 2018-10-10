%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course:  Understanding Deep Neural Networks
% Teacher: Zhang Yi
% Student:³ÂÏé
% ID:2016141482081
%
% Lab 4 - Example: Handwritten Digit Recognition
% Files:
% 1. lab4.m
% 2. fc.m
% 3. bc.m
%
% Tasks:
% 1. recognize handwritten digits in images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a_next, z_next] = fc(w, a, x)
    % define the activation function
    f = @(s) 1 ./ (1 + exp(-s));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % forward computing (either component or vector form)
    a = [x
        a];
    z_next = w * a;
    a_next = f(z_next);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
