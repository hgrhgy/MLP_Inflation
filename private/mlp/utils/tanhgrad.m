function g = tanhgrad(z)

%this script returns the gradient of the sigmoid function evaluated at z
g=1-tanh(z).*tanh(z)

end
