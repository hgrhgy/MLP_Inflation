function g = relugrad(z)

%this script returns the gradient of the sigmoid function evaluated at z
    if (z >= 0)
        g = 1;
    else
        g = 0.1;
    end

end
