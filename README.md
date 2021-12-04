# Solver for Two-Dimensional Linear Programs

* Single C++20 file.

* Solves problems on the form
```
min   y,
s.t.  ai x + bi * y <= ci.
```

* Based on [Megiddo's algorithm](https://doi.org/10.1109/SFCS.1982.24)

## Example

```cpp
std::vector<std::array<double, 3>> lp{
  {0., -1., 2.},    // y >= -2
  {0., -1., 1.5},   // y >= -1.5
  {-1., -1., 0.},   // y >= -x (*)
  {-1., -1., 0.2},  // y >= -x - 0.2
  {1., -1., 2.},    // y >= x - 2 (*)
};

const auto [xopt, yopt] = lp2d::solve(lp);
```
