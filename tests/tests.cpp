// lp2d: Two-Dimensional Linear Programming
// https://github.com/pettni/lp2d
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2021 Petter Nilsson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <catch2/catch.hpp>
#include <lp2d/lp2d.hpp>

#include <random>
#include <vector>

TEST_CASE("Basic")
{
  std::vector<std::array<double, 3>> rows{
    {0., -1., 2.},    // y >= -2
    {0., -1., 1.5},   // y >= -1.5
    {-1., -1., 0.},   // y >= -x (*)
    {-1., -1., 0.2},  // y >= -x - 0.2
    {1., -1., 2.},    // y >= x - 2 (*)
  };

  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, rows);

  REQUIRE(xopt == Approx(1).epsilon(1e-9));
  REQUIRE(yopt == Approx(-1).epsilon(1e-9));
}

TEST_CASE("BasicFlipped")
{
  std::vector<std::array<double, 3>> rows{
    {0., 1., 2.},    // y >= -2
    {0., 1., 1.5},   // y >= -1.5
    {-1., 1., 0.},   // y >= -x (*)
    {-1., 1., 0.2},  // y >= -x - 0.2
    {1., 1., 2.},    // y >= x - 2 (*)
  };

  const auto [xopt, yopt, stat] = lp2d::solve(0, -1, rows);

  REQUIRE(xopt == Approx(1).epsilon(1e-9));
  REQUIRE(yopt == Approx(1).epsilon(1e-9));
}

TEST_CASE("BasicRotated")
{
  std::vector<std::array<double, 3>> rows{
    {-1., 0., 2.},
    {-1., 0., 1.5},
    {-1., -1., 0.},
    {-1., -1., 0.2},
    {-1., 1., 2.},
  };

  const auto [xopt, yopt, stat] = lp2d::solve(1, 0, rows);

  REQUIRE(xopt == Approx(-1).epsilon(1e-9));
  REQUIRE(yopt == Approx(1).epsilon(1e-9));
}

TEST_CASE("Empty")
{
  std::vector<std::array<double, 3>> hps{};
  {
    const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
    REQUIRE(stat == lp2d::Status::DualInfeasible);
  }
  {
    const auto [xopt, yopt, stat] = lp2d::solve(1, 1, hps);
    REQUIRE(stat == lp2d::Status::DualInfeasible);
  }
  {
    const auto [xopt, yopt, stat] = lp2d::solve(-100, 1, hps);
    REQUIRE(stat == lp2d::Status::DualInfeasible);
  }
}

TEST_CASE("SingleLowerFlat")
{
  std::vector<std::array<double, 3>> hps{{0, -1, 2}, {1, 0, 3}, {-1, 0, 3}};
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(yopt == Approx(-2).epsilon(1e-9));
}

TEST_CASE("SingleLowerFlatBounds")
{
  std::vector<std::array<double, 3>> hps{
    {0, -1, 2},
    {1, 0, 1},
    {-1, 0, 1},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(yopt == Approx(-2).epsilon(1e-9));
}

TEST_CASE("SingleLowerTilted")
{
  std::vector<std::array<double, 3>> hps{{0.001, -1, 2}};
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::DualInfeasible);
}

TEST_CASE("SingleLowerTiltedBounds")
{
  std::vector<std::array<double, 3>> hps{
    {0.001, -1, 2},
    {1, 0, 1},
    {-1, 0, 1},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(yopt == Approx(-2.001).epsilon(1e-9));
}

TEST_CASE("Bounds")
{
  std::vector<std::array<double, 3>> hps{
    {1, 0, 1},
    {-1, 0, 1},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::DualInfeasible);
}

TEST_CASE("SingleUpperFlat")
{
  std::vector<std::array<double, 3>> hps{{0, 1, 2}};
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::DualInfeasible);
}

TEST_CASE("SingleUpperTilted")
{
  std::vector<std::array<double, 3>> hps{{0.2, 1, 2}};
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::DualInfeasible);
}

TEST_CASE("UpperLowerIsect")
{
  std::vector<std::array<double, 3>> hps{
    {-1, 1, -2},
    {1, -4, 9},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(yopt == -7. / 3);
}

TEST_CASE("UpperLowerParInfeas")
{
  std::vector<std::array<double, 3>> hps{
    {-1, 4, -3},
    {1, -4, 2},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::PrimaryInfeasible);
}

TEST_CASE("UpperLowerParInfeasBounds")
{
  std::vector<std::array<double, 3>> hps{
    {-1, 4, -3},
    {1, -4, 2},
    {1, 0, 1},
    {-1, 0, 1},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::PrimaryInfeasible);
}

TEST_CASE("UpperLowerParFeas")
{
  std::vector<std::array<double, 3>> hps{
    {-1, 4, -1},
    {1, -4, 2},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::DualInfeasible);
}

TEST_CASE("UpperLowerParFeasBounds")
{
  std::vector<std::array<double, 3>> hps{
    {-1, 4, -1},
    {1, -4, 2},
    {1, 0, 1},
    {-1, 0, 1},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(yopt == Approx(-0.75).epsilon(1e-9));
}

TEST_CASE("UpperLowerParInfeasFlat")
{
  std::vector<std::array<double, 3>> hps{
    {0, 4, -3},
    {0, -4, 2},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::PrimaryInfeasible);
}

TEST_CASE("UpperLowerParInfeasFlatBounds")
{
  std::vector<std::array<double, 3>> hps{
    {0, 4, -3},
    {0, -4, 2},
    {1, 0, 1},
    {-1, 0, 1},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::PrimaryInfeasible);
}

TEST_CASE("UpperLowerParFeasFlat")
{
  std::vector<std::array<double, 3>> hps{
    {0, 4, -1},
    {0, -4, 2},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(yopt == Approx(-1. / 2).epsilon(1e-9));
}

TEST_CASE("Diamond")
{
  std::vector<std::array<double, 3>> hps{
    {1, 1, 1},   {-1, 1, 1},  {1, -1, 1},  {-1, -1, 1}, {1, 1, 1},   {-1, 1, 1},   {1, -1, 1},
    {-1, -1, 1}, {1, 1, 2},   {-1, 1, 2},  {1, -1, 2},  {-1, -1, 2}, {1, 1., 0.5}, {1, -1., 1.2},
    {1, 1., 1},  {1, 1., 2},  {1, -1., 3}, {1, 1., 3},  {1, -1., 4}, {1, 1., 4},   {1, -1., 5},
    {1, 1., 5},  {1, -1., 6}, {1, 1., 6},  {1, 1., 7},  {1, -1., 7},
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(yopt == Approx(-1.).epsilon(1e-9));
}

TEST_CASE("SinglePoint")
{
  std::vector<std::array<double, 3>> hps{
    {0, -1, 1},   // y >= -1
    {-1, 1, -2},  // y <= -2 + x
    {1, 1, 0},    // y <=  -x
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::Optimal);
  REQUIRE(yopt == Approx(-1).epsilon(1e-9));
}

TEST_CASE("Infeas")
{
  std::vector<std::array<double, 3>> hps{
    {0, -1, -0.9},  // y >= -0.9
    {-1, 1, -2},    // y <= -2 + x
    {1, 1, 0},      // y <=  -x
  };
  const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);
  REQUIRE(stat == lp2d::Status::PrimaryInfeasible);
}

TEST_CASE("Random")
{
  std::default_random_engine rng(5);

  for (auto iter = 0u; iter < 100; ++iter) {
    std::vector<std::array<double, 3>> hps;

    std::uniform_real_distribution<double> distr(0, 1);

    for (auto i = 0u; i < 25; ++i) {
      hps.push_back(std::array<double, 3>{
        2 * (distr(rng) - 0.5),
        2 * (distr(rng) - 0.5),
        distr(rng),
      });
    }

    const auto [xopt, yopt, stat] = lp2d::solve(0, 1, hps);

    REQUIRE(stat == lp2d::Status::Optimal);

    // check feasible
    for (const auto [ax, ay, b] : hps) {
      REQUIRE(ax * xopt + ay * yopt <= Approx(b).epsilon(1e-9));
    }
  }
}
