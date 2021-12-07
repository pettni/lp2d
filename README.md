# Solver for Two-Dimensional Linear Programs

* Single C++20 file.

* Solves 2D linear optimization problems on the form
```
min_{x,y}   cx  * x +  cy * y,
s.t.        axi * x + ayi * y <= bi.
```

* Based on [Megiddo's algorithm](https://doi.org/10.1109/SFCS.1982.24)

## Example

```cpp
std::vector<std::array<double, 3>> rows{
  {0., -1., 2.},    //    - y <= 2
  {0., -1., 1.5},   //    - y <= 1.5
  {-1., -1., 0.},   // -x - y <= 0 
  {-1., -1., 0.2},  // -x - y <= 0.2
  {1., -1., 2.},    //  x - y <= 2
};

const auto [xopt, yopt] = lp2d::solve(0, 1, rows);
```
