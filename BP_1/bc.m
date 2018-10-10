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

function delta = bc(w, z, delta_next)
    % define the activation function
    f = @(s) 1 ./ (1 + exp(-s)); 
    % define the derivative of activation function
    df = @(s) f(s) .* (1 - f(s)); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % backward computing (either component or vector form)
    
    k = size(z, 1);
    delta = w' * delta_next;
    delta = delta(1: k, : ) .* df(z);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end