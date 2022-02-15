using Roots

f(x) = exp(x)-x^4
cand = 8
fx = ZeroProblem(f,cand)

x0 = find_zero(f,(8,9))

x1 = solve(fx)
println("One root is $x0")
println("Another method yields to the root $x1")