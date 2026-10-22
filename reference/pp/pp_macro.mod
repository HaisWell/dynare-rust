// macro language: arrays, tuples, comprehensions, macro functions, @#for with tuples/when (stream PP)
@#define countries = ["fr", "de", "it"]
@#define weights = [0.5, 0.3, 0.2]
@#define pairs = [("fr", 1), ("de", 2), ("it", 3)]
@#define big = [c for c in countries when c != "it"]
@#define sq = [i^2 for i in 1:4]
@#define f(x, y) = 2*x + y
@#define lagged(v, k) = v + "(-" + k + ")"
var
@#for c in countries
  y_@{c}
@#endfor
  ybar;
varexo
@#for (c, k) in pairs
  e_@{c}
@#endfor
;
parameters rho;
rho = @{f(0.2, 0.1)};
model;
@#for (c, k) in pairs
y_@{c} = rho*@{lagged("y_" + c, "1")} + @{k/10}*e_@{c};
@#endfor
ybar = 0
@#for i in 1:3
  + @{weights[i]}*y_@{countries[i]}
@#endfor
@#for c in big when length(c) == 2
  + 0*y_@{c}(+@{sq[2]})
@#endfor
;
end;
shocks;
@#for c in countries
var e_@{c}; stderr 0.01;
@#endfor
end;
stoch_simul(order=1, irf=0, nograph);
