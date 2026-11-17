// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"
#include "options.h"

namespace tinyopt::traits {

template <>
struct params_trait<tinyopt::benchmark::bundle_adjustment::Problem> {
  using Scalar = double;
  static constexpr Index Dims = Dynamic;

  static Index dims(const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    return 6 * problem.CameraCount() + 3 * problem.PointCount();
  }

  static std::pair<Index, Index> reduced(
      const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    return {0, 6 * problem.CameraCount()};
  }

  static std::vector<Index> fixed(const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    const Index camera_offset = 6 * problem.CameraCount();
    return {0, 1, 2, 3, 4, 5, camera_offset, camera_offset + 1, camera_offset + 2};
  }

  static void plus_eq(tinyopt::benchmark::bundle_adjustment::Problem& problem, const auto& delta) {
    const int camera_offset = 6 * problem.CameraCount();
    for (int camera = 0; camera < problem.CameraCount(); ++camera)
      problem.initial_cameras[camera].plus_eq(delta.template segment<6>(camera * 6));
    for (int point = 0; point < problem.PointCount(); ++point)
      problem.initial_points[point] += delta.template segment<3>(camera_offset + point * 3);
  }

  static Scalar norm2(const tinyopt::benchmark::bundle_adjustment::Problem& problem) {
    Scalar sum = 0.0;
    for (const auto& camera : problem.initial_cameras) sum += camera.params.squaredNorm();
    for (const auto& point : problem.initial_points) sum += point.squaredNorm();
    return sum;
  }
};

}  // namespace tinyopt::traits

namespace tinyopt::benchmark::bundle_adjustment {

// Tinyopt options matching the Ceres/g2o bundle-adjustment settings: Jacobi scaling and a relative
// parameter-step tolerance ((1e-8)^2 on the squared relative step, like Ceres'
// parameter_tolerance).
inline Options CreateBundleAdjustmentOptions() {
  Options options = CreateOptions();
  options.lm.jacobi_scaling = true;
  options.stop.min_rerr_dec = static_cast<float>(RelativeCostDecreaseTol);
  options.stop.min_step_norm2 = 0.0;
  options.stop.min_step_rel_norm2 = 1e-16;
  return options;
}

struct BlockSparseBundleAdjustmentLoss {
  using Hessian = BlockSparseMatrix<double, 6, 3>;

  void set_num_threads(std::uint16_t num_threads) const { num_threads_ = num_threads; }

  Cost operator()(const Problem& problem, auto& gradient, Hessian& hessian) const {
    const int camera_offset = 6 * problem.CameraCount();
    const int threads = parallel::ThreadCount(num_threads_);
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      if (threads > 1 && problem.observations.size() >= 256) {
        PrepareEvaluations(problem);
        PrepareHessianStructure(problem, hessian);

        parallel::ParallelFor(problem.observations.size(), 256, num_threads_,
                               [&](std::size_t index) {
                                 EvaluateObservation(problem, problem.observations[index],
                                                     evaluations_[index]);
                               });

        double squared_error = 0.0;
        for (std::size_t index = 0; index < problem.observations.size(); ++index) {
          squared_error += evaluations_[index].residual.squaredNorm();
        }
        parallel::ParallelFor(problem.CameraCount(), 1, num_threads_, [&](std::size_t camera) {
          for (const std::size_t index : camera_observations_[camera]) {
            AccumulateCameraEvaluation(problem.observations[index], evaluations_[index], gradient,
                                       hessian);
          }
        });
        parallel::ParallelFor(problem.PointCount(), 1, num_threads_, [&](std::size_t point) {
          for (const std::size_t index : point_observations_[point]) {
            AccumulatePointEvaluation(problem.observations[index], evaluations_[index],
                                      camera_offset, gradient, hessian);
          }
        });
        return Cost(0.5 * squared_error,
                    static_cast<int>(2 * problem.observations.size()));
      }
    }

    return AccumulateSerial(problem, camera_offset, gradient, hessian);
  }

 private:
  struct ObservationEvaluation {
    Eigen::Matrix<double, 2, 6> camera_jacobian;
    Eigen::Matrix<double, 2, 3> point_jacobian;
    ImagePoint residual;
  };

  void PrepareEvaluations(const Problem& problem) const {
    if (evaluations_.size() == problem.observations.size() &&
        camera_observations_.size() == static_cast<std::size_t>(problem.CameraCount()))
      return;
    evaluations_.resize(problem.observations.size());
    camera_observations_.assign(static_cast<std::size_t>(problem.CameraCount()), {});
    point_observations_.assign(static_cast<std::size_t>(problem.PointCount()), {});
    for (std::size_t index = 0; index < problem.observations.size(); ++index) {
      const auto& observation = problem.observations[index];
      camera_observations_[static_cast<std::size_t>(observation.camera)].push_back(index);
      point_observations_[static_cast<std::size_t>(observation.point)].push_back(index);
    }
  }

  static void PrepareHessianStructure(const Problem& problem, Hessian& hessian) {
    for (int camera = 0; camera < problem.CameraCount(); ++camera) {
      if (!hessian.template has_block<6, 6>(camera, camera))
        hessian.template block<6, 6>(camera, camera).setZero();
    }
    for (int point = 0; point < problem.PointCount(); ++point) {
      if (!hessian.template has_block<3, 3>(point, point))
        hessian.template block<3, 3>(point, point).setZero();
    }
    for (const auto& observation : problem.observations) {
      if (!hessian.template has_block<6, 3>(observation.camera, observation.point))
        hessian.template block<6, 3>(observation.camera, observation.point).setZero();
    }
  }

  static void EvaluateObservation(const Problem& problem, const Observation& observation,
                                  ObservationEvaluation& evaluation) {
    const Camera& camera = problem.initial_cameras[observation.camera];
    const Point& point = problem.initial_points[observation.point];
    ImagePoint projected;
    ProjectionJacobians(camera, point, projected, evaluation.camera_jacobian,
                        evaluation.point_jacobian);
    evaluation.residual = projected - observation.measurement;
  }

  static void AccumulateEvaluation(const Observation& observation,
                                   const ObservationEvaluation& evaluation, int camera_offset,
                                   Vector<double, Dynamic>& gradient, Hessian& hessian) {
    const Index camera_start = observation.camera * 6;
    const Index point_start = camera_offset + observation.point * 3;
    gradient.template segment<6>(camera_start).noalias() +=
        evaluation.camera_jacobian.transpose() * evaluation.residual;
    gradient.template segment<3>(point_start).noalias() +=
        evaluation.point_jacobian.transpose() * evaluation.residual;
    hessian.template block<6, 6>(observation.camera, observation.camera).noalias() +=
        evaluation.camera_jacobian.transpose() * evaluation.camera_jacobian;
    hessian.template block<3, 3>(observation.point, observation.point).noalias() +=
        evaluation.point_jacobian.transpose() * evaluation.point_jacobian;
    hessian.template block<6, 3>(observation.camera, observation.point).noalias() +=
        evaluation.camera_jacobian.transpose() * evaluation.point_jacobian;
  }

  static void AccumulateCameraEvaluation(const Observation& observation,
                                         const ObservationEvaluation& evaluation,
                                         Vector<double, Dynamic>& gradient, Hessian& hessian) {
    const Index camera_start = observation.camera * 6;
    gradient.template segment<6>(camera_start).noalias() +=
        evaluation.camera_jacobian.transpose() * evaluation.residual;
    hessian.template block<6, 6>(observation.camera, observation.camera).noalias() +=
        evaluation.camera_jacobian.transpose() * evaluation.camera_jacobian;
    hessian.template block<6, 3>(observation.camera, observation.point).noalias() +=
        evaluation.camera_jacobian.transpose() * evaluation.point_jacobian;
  }

  static void AccumulatePointEvaluation(const Observation& observation,
                                        const ObservationEvaluation& evaluation, int camera_offset,
                                        Vector<double, Dynamic>& gradient, Hessian& hessian) {
    const Index point_start = camera_offset + observation.point * 3;
    gradient.template segment<3>(point_start).noalias() +=
        evaluation.point_jacobian.transpose() * evaluation.residual;
    hessian.template block<3, 3>(observation.point, observation.point).noalias() +=
        evaluation.point_jacobian.transpose() * evaluation.point_jacobian;
  }

  static Cost AccumulateSerial(const Problem& problem, int camera_offset, auto& gradient,
                               Hessian& hessian) {
    double squared_error = 0.0;
    for (const auto& observation : problem.observations) {
      if constexpr (traits::is_nullptr_v<decltype(gradient)>) {
        const ImagePoint residual =
            Project(problem.initial_cameras[observation.camera],
                    problem.initial_points[observation.point]) -
            observation.measurement;
        squared_error += residual.squaredNorm();
      } else {
        Eigen::Matrix<double, 2, 6> camera_jacobian;
        Eigen::Matrix<double, 2, 3> point_jacobian;
        ImagePoint projected;
        ProjectionJacobians(problem.initial_cameras[observation.camera],
                            problem.initial_points[observation.point], projected, camera_jacobian,
                            point_jacobian);
        const ImagePoint residual = projected - observation.measurement;
        squared_error += residual.squaredNorm();
        const ObservationEvaluation evaluation{camera_jacobian, point_jacobian, residual};
        AccumulateEvaluation(observation, evaluation, camera_offset, gradient, hessian);
      }
    }
    return Cost(0.5 * squared_error, static_cast<int>(2 * problem.observations.size()));
  }

  mutable std::uint16_t num_threads_ = 1;
  mutable std::vector<ObservationEvaluation> evaluations_;
  mutable std::vector<std::vector<std::size_t>> camera_observations_;
  mutable std::vector<std::vector<std::size_t>> point_observations_;
};

using BlockSparseLoss = BlockSparseBundleAdjustmentLoss;
using TinyoptLoss = BlockSparseBundleAdjustmentLoss;

struct SparseBundleAdjustmentLoss {
  using Hessian = SparseMat;

  Cost operator()(const Problem& problem, auto& gradient, Hessian& hessian) const {
    const int camera_offset = 6 * problem.CameraCount();
    double squared_error = 0.0;

    Eigen::Matrix<double, 2, 6> camera_jacobian;
    Eigen::Matrix<double, 2, 3> point_jacobian;
    ImagePoint projected;

    std::vector<Eigen::Triplet<double>> triplets;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      triplets.reserve(problem.observations.size() * (36 + 9 + 36));
    }

    for (const auto& observation : problem.observations) {
      const Camera& camera = problem.initial_cameras[observation.camera];
      const Point& point = problem.initial_points[observation.point];

      if constexpr (traits::is_nullptr_v<decltype(gradient)>) {
        const ImagePoint residual = Project(camera, point) - observation.measurement;
        squared_error += residual.squaredNorm();
      } else {
        ProjectionJacobians(camera, point, projected, camera_jacobian, point_jacobian);
        const ImagePoint residual = projected - observation.measurement;
        squared_error += residual.squaredNorm();

        const Index camera_block = observation.camera;
        const Index point_block = observation.point;
        const Index camera_start = camera_block * 6;
        const Index point_start = camera_offset + point_block * 3;

        gradient.template segment<6>(camera_start).noalias() +=
            camera_jacobian.transpose() * residual;
        gradient.template segment<3>(point_start).noalias() +=
            point_jacobian.transpose() * residual;

        const Eigen::Matrix<double, 6, 6> JtJ_cam = camera_jacobian.transpose() * camera_jacobian;
        for (int r = 0; r < 6; ++r) {
          for (int c = 0; c < 6; ++c) {
            triplets.emplace_back(camera_start + r, camera_start + c, JtJ_cam(r, c));
          }
        }

        const Eigen::Matrix<double, 3, 3> JtJ_pt = point_jacobian.transpose() * point_jacobian;
        for (int r = 0; r < 3; ++r) {
          for (int c = 0; c < 3; ++c) {
            triplets.emplace_back(point_start + r, point_start + c, JtJ_pt(r, c));
          }
        }

        const Eigen::Matrix<double, 6, 3> JtJ_cp = camera_jacobian.transpose() * point_jacobian;
        for (int r = 0; r < 6; ++r) {
          for (int c = 0; c < 3; ++c) {
            const double val = JtJ_cp(r, c);
            triplets.emplace_back(camera_start + r, point_start + c, val);
            triplets.emplace_back(point_start + c, camera_start + r, val);
          }
        }
      }
    }

    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      hessian.setFromTriplets(triplets.begin(), triplets.end());
    }

    return Cost(0.5 * squared_error, static_cast<int>(2 * problem.observations.size()));
  }
};

using SparseLoss = SparseBundleAdjustmentLoss;

}  // namespace tinyopt::benchmark::bundle_adjustment