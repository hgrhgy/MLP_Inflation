function g = tanh(x)

% returns sigmoid evaluated elementwize in X
    g = (exp(x)-exp(-x)) ./ (exp(x)+exp(-x)); 

end