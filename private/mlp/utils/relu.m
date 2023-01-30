function g = relu(x)

% returns sigmoid evaluated elementwize in X
    if (x>=0)
        g = x;
    else
        g = 0.1 * x;
    end
end