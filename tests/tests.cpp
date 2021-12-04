#include "lp2d/lp2d.hpp"
#include <catch2/catch.hpp>

#include <iostream>
#include <random>
#include <vector>

TEST_CASE("Basic")
{
  std::vector<std::array<double, 3>> lp{
    {0., -1., 2.},    // y >= -2
    {0., -1., 1.5},   // y >= -1.5
    {-1., -1., 0.},   // y >= -x (*)
    {-1., -1., 0.2},  // y >= -x - 0.2
    {1., -1., 2.},    // y >= x - 2 (*)
  };

  const auto [xopt, yopt] = lp2d::solve(lp);

  REQUIRE(yopt == Approx(-1).epsilon(1e-9));
}

TEST_CASE("Empty")
{
  std::vector<lp2d::detail::HalfPlane> hps{};
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == -lp2d::detail::inf);
}

TEST_CASE("SingleLowerFlat")
{
  std::vector<lp2d::detail::HalfPlane> hps{{.a = 0, .b = -1, .c = 2}};
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == Approx(-2).epsilon(1e-9));
}

TEST_CASE("SingleLowerFlatBounds")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 0, .b = -1, .c = 2},
    {.a = 1, .b = 0, .c = 1},
    {.a = -1, .b = 0, .c = 1},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == Approx(-2).epsilon(1e-9));
}

TEST_CASE("SingleLowerTilted")
{
  std::vector<lp2d::detail::HalfPlane> hps{{.a = 0.001, .b = -1, .c = 2}};
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == -lp2d::detail::inf);
}

TEST_CASE("SingleLowerTiltedBounds")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 0.001, .b = -1, .c = 2},
    {.a = 1, .b = 0, .c = 1},
    {.a = -1, .b = 0, .c = 1},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == Approx(-2.001).epsilon(1e-9));
}

TEST_CASE("Bounds")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 1, .b = 0, .c = 1},
    {.a = -1, .b = 0, .c = 1},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == -lp2d::detail::inf);
}

TEST_CASE("SingleUpperFlat")
{
  std::vector<lp2d::detail::HalfPlane> hps{{.a = 0, .b = 1, .c = 2}};
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == -lp2d::detail::inf);
}

TEST_CASE("SingleUpperTilted")
{
  std::vector<lp2d::detail::HalfPlane> hps{{.a = 0.2, .b = 1, .c = 2}};
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == -lp2d::detail::inf);
}

TEST_CASE("UpperLowerIsect")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = -1, .b = 1, .c = -2},
    {.a = 1, .b = -4, .c = 9},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(lp2d::detail::check(hps, xopt) == 0);
  REQUIRE(yopt == -7. / 3);
}

TEST_CASE("UpperLowerParInfeas")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = -1, .b = 4, .c = -3},
    {.a = 1, .b = -4, .c = 2},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == lp2d::detail::inf);
}

TEST_CASE("UpperLowerParInfeasBounds")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = -1, .b = 4, .c = -3},
    {.a = 1, .b = -4, .c = 2},
    {.a = 1, .b = 0, .c = 1},
    {.a = -1, .b = 0, .c = 1},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == lp2d::detail::inf);
}

TEST_CASE("UpperLowerParFeas")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = -1, .b = 4, .c = -1},
    {.a = 1, .b = -4, .c = 2},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == -lp2d::detail::inf);
}

TEST_CASE("UpperLowerParFeasBounds")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = -1, .b = 4, .c = -1},
    {.a = 1, .b = -4, .c = 2},
    {.a = 1, .b = 0, .c = 1},
    {.a = -1, .b = 0, .c = 1},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == Approx(-0.75).epsilon(1e-9));
}

TEST_CASE("UpperLowerParInfeasFlat")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 0, .b = 4, .c = -3},
    {.a = 0, .b = -4, .c = 2},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == lp2d::detail::inf);
}

TEST_CASE("UpperLowerParInfeasFlatBounds")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 0, .b = 4, .c = -3},
    {.a = 0, .b = -4, .c = 2},
    {.a = 1, .b = 0, .c = 1},
    {.a = -1, .b = 0, .c = 1},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == lp2d::detail::inf);
}

TEST_CASE("UpperLowerParFeasFlat")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 0, .b = 4, .c = -1},
    {.a = 0, .b = -4, .c = 2},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == Approx(-1. / 2).epsilon(1e-9));
}

TEST_CASE("Diamond")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 1, .b = 1, .c = 1},    {.a = -1, .b = 1, .c = 1},    {.a = 1, .b = -1, .c = 1},
    {.a = -1, .b = -1, .c = 1},  {.a = 1, .b = 1, .c = 1},     {.a = -1, .b = 1, .c = 1},
    {.a = 1, .b = -1, .c = 1},   {.a = -1, .b = -1, .c = 1},   {.a = 1, .b = 1, .c = 2},
    {.a = -1, .b = 1, .c = 2},   {.a = 1, .b = -1, .c = 2},    {.a = -1, .b = -1, .c = 2},
    {.a = 1, .b = 1., .c = 0.5}, {.a = 1, .b = -1., .c = 1.2}, {.a = 1, .b = 1., .c = 1},
    {.a = 1, .b = 1., .c = 2},   {.a = 1, .b = -1., .c = 3},   {.a = 1, .b = 1., .c = 3},
    {.a = 1, .b = -1., .c = 4},  {.a = 1, .b = 1., .c = 4},    {.a = 1, .b = -1., .c = 5},
    {.a = 1, .b = 1., .c = 5},   {.a = 1, .b = -1., .c = 6},   {.a = 1, .b = 1., .c = 6},
    {.a = 1, .b = 1., .c = 7},   {.a = 1, .b = -1., .c = 7},
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(lp2d::detail::check(hps, xopt) == 0);
  REQUIRE(yopt == Approx(-1.).epsilon(1e-9));
}

TEST_CASE("SinglePoint")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 0, .b = -1, .c = 1},   // y >= -1
    {.a = -1, .b = 1, .c = -2},  // y <= -2 + x
    {.a = 1, .b = 1, .c = 0},    // y <=  -x
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(lp2d::detail::check(hps, xopt) == 0);
  REQUIRE(yopt == Approx(-1).epsilon(1e-9));
}

TEST_CASE("Infeas")
{
  std::vector<lp2d::detail::HalfPlane> hps{
    {.a = 0, .b = -1, .c = -0.9},  // y >= -0.9
    {.a = -1, .b = 1, .c = -2},    // y <= -2 + x
    {.a = 1, .b = 1, .c = 0},      // y <=  -x
  };
  const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);
  REQUIRE(yopt == lp2d::detail::inf);
}

TEST_CASE("Random")
{
  std::default_random_engine rng(5);

  for (auto iter = 0u; iter < 100; ++iter) {
    std::vector<lp2d::detail::HalfPlane> hps;

    std::uniform_real_distribution<double> distr(0, 1);

    for (auto i = 0u; i < 25; ++i) {
      hps.push_back(lp2d::detail::HalfPlane{
        .a = 2 * (distr(rng) - 0.5),
        .b = 2 * (distr(rng) - 0.5),
        .c = distr(rng),
      });
    }

    const auto hps_copy     = hps;
    const auto [xopt, yopt] = lp2d::detail::solve_impl(hps);

    // check feasible
    for (const auto hp : hps_copy) {
      REQUIRE(hp.a * xopt + hp.b * yopt <= Approx(hp.c).epsilon(1e-9));
      REQUIRE(lp2d::detail::check(hps, xopt) == 0);
      REQUIRE(lp2d::detail::check(hps_copy, xopt) == 0);
    }
  }
}
