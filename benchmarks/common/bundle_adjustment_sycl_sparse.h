// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "bundle_adjustment.h"
#include "bundle_adjustment_tinyopt.h"

namespace tinyopt::benchmark::bundle_adjustment_sycl {

namespace sparse_detail {

struct CameraData {
  double rotation[9]{};
  double translation[3]{};
};

struct PointData {
  double position[3]{};
};

struct ObservationData {
  int camera = 0;
  int point = 0;
  int coupling_block = -1;
  double measurement[2]{};
};

struct ObservationEvaluation {
  double cost = 0.0;
  double residual[2]{};
  double camera_jacobian[12]{};
  double point_jacobian[6]{};
};

struct HostData {
  std::vector<CameraData> cameras;
  std::vector<PointData> points;
  std::vector<ObservationData> observations;
  std::vector<std::size_t> camera_offsets;
  std::vector<std::size_t> camera_observations;
  std::vector<std::size_t> point_offsets;
  std::vector<std::size_t> point_observations;
  std::size_t coupling_block_count = 0;
};

template <typename T>
void CopyToBuffer(sycl::buffer<T, 1>& buffer, const std::vector<T>& values) {
  auto output = buffer.get_host_access();
  for (std::size_t i = 0; i < values.size(); ++i) output[i] = values[i];
}

inline HostData PrepareHostData(const bundle_adjustment::Problem& problem,
                                bundle_adjustment::BlockSparseLoss::Hessian& hessian) {
  HostData data;
  data.cameras.resize(static_cast<std::size_t>(problem.CameraCount()));
  data.points.resize(static_cast<std::size_t>(problem.PointCount()));

  for (int camera_index = 0; camera_index < problem.CameraCount(); ++camera_index) {
    const auto rotation = problem.initial_cameras[static_cast<std::size_t>(camera_index)]
                              .rotation()
                              .toRotationMatrix();
    const auto translation =
        problem.initial_cameras[static_cast<std::size_t>(camera_index)].translation();
    CameraData& camera = data.cameras[static_cast<std::size_t>(camera_index)];
    for (int row = 0; row < 3; ++row) {
      camera.translation[row] = translation[row];
      for (int col = 0; col < 3; ++col)
        camera.rotation[col * 3 + row] = rotation(row, col);
    }
  }

  for (int point_index = 0; point_index < problem.PointCount(); ++point_index) {
    const auto& position = problem.initial_points[static_cast<std::size_t>(point_index)];
    for (int axis = 0; axis < 3; ++axis)
      data.points[static_cast<std::size_t>(point_index)].position[axis] = position[axis];
  }

  for (int camera = 0; camera < problem.CameraCount(); ++camera)
    if (!hessian.template has_block<6, 6>(camera, camera))
      hessian.template block<6, 6>(camera, camera).setZero();
  for (int point = 0; point < problem.PointCount(); ++point)
    if (!hessian.template has_block<3, 3>(point, point))
      hessian.template block<3, 3>(point, point).setZero();

  data.observations.reserve(problem.observations.size());
  for (const auto& observation : problem.observations) {
    if (!hessian.template has_block<6, 3>(observation.camera, observation.point))
      hessian.template block<6, 3>(observation.camera, observation.point).setZero();
  }
  data.coupling_block_count = hessian.template blocks<6, 3>().size();
  std::vector<bool> written(data.coupling_block_count, false);
  for (const auto& observation : problem.observations) {
    const int block = hessian.template block_index<6, 3>(observation.camera, observation.point);
    if (block < 0) throw std::logic_error("Missing BA camera-point Hessian block");
    if (written[static_cast<std::size_t>(block)])
      throw std::invalid_argument("SYCL BA benchmark requires unique camera-point observations");
    written[static_cast<std::size_t>(block)] = true;
    data.observations.push_back({observation.camera, observation.point, block,
                                 {observation.measurement[0], observation.measurement[1]}});
  }
  if (std::find(written.begin(), written.end(), false) != written.end())
    throw std::invalid_argument("SYCL BA benchmark requires a Hessian matching the observation graph");

  std::vector<std::vector<std::size_t>> camera_rows(data.cameras.size());
  std::vector<std::vector<std::size_t>> point_rows(data.points.size());
  for (std::size_t index = 0; index < data.observations.size(); ++index) {
    const auto& observation = data.observations[index];
    camera_rows[static_cast<std::size_t>(observation.camera)].push_back(index);
    point_rows[static_cast<std::size_t>(observation.point)].push_back(index);
  }

  data.camera_offsets.reserve(data.cameras.size() + 1);
  data.camera_offsets.push_back(0);
  for (const auto& row : camera_rows) {
    data.camera_observations.insert(data.camera_observations.end(), row.begin(), row.end());
    data.camera_offsets.push_back(data.camera_observations.size());
  }
  data.point_offsets.reserve(data.points.size() + 1);
  data.point_offsets.push_back(0);
  for (const auto& row : point_rows) {
    data.point_observations.insert(data.point_observations.end(), row.begin(), row.end());
    data.point_offsets.push_back(data.point_observations.size());
  }
  return data;
}

}  // namespace sparse_detail

/// GPU accumulation for the repository's camera/point block-sparse BA benchmark.
///
/// Observation residuals and Jacobians are evaluated on the device. Camera and point diagonal
/// blocks are reduced there, and each unique camera-point coupling block is produced there. Only
/// the resulting sparse blocks are copied into Tinyopt's host-side BlockSparseMatrix.
class SyclBlockSparseAccumulator {
 public:
  using Problem = bundle_adjustment::Problem;
  using Hessian = bundle_adjustment::BlockSparseLoss::Hessian;

  SyclBlockSparseAccumulator(sycl::queue& queue, const Problem& problem, Hessian& hessian)
      : queue_(queue),
        hessian_(hessian),
        host_data_(sparse_detail::PrepareHostData(problem, hessian)),
        camera_buffer_(sycl::range<1>(host_data_.cameras.size())),
        point_buffer_(sycl::range<1>(host_data_.points.size())),
        observation_buffer_(sycl::range<1>(host_data_.observations.size())),
        camera_offsets_buffer_(sycl::range<1>(host_data_.camera_offsets.size())),
        camera_observations_buffer_(sycl::range<1>(host_data_.camera_observations.size())),
        point_offsets_buffer_(sycl::range<1>(host_data_.point_offsets.size())),
        point_observations_buffer_(sycl::range<1>(host_data_.point_observations.size())),
        evaluation_buffer_(sycl::range<1>(host_data_.observations.size())),
        coupling_buffer_(sycl::range<1>(host_data_.coupling_block_count * 18)),
        diagonal_buffer_(sycl::range<1>(host_data_.cameras.size() * 42 +
                                         host_data_.points.size() * 12)),
        cost_buffer_(sycl::range<1>(host_data_.observations.size())),
        cost_partials_buffer_(sycl::range<1>(CostPartialCount())),
        total_cost_buffer_(sycl::range<1>(1)) {
    sparse_detail::CopyToBuffer(camera_buffer_, host_data_.cameras);
    sparse_detail::CopyToBuffer(point_buffer_, host_data_.points);
    sparse_detail::CopyToBuffer(observation_buffer_, host_data_.observations);
    sparse_detail::CopyToBuffer(camera_offsets_buffer_, host_data_.camera_offsets);
    sparse_detail::CopyToBuffer(camera_observations_buffer_, host_data_.camera_observations);
    sparse_detail::CopyToBuffer(point_offsets_buffer_, host_data_.point_offsets);
    sparse_detail::CopyToBuffer(point_observations_buffer_, host_data_.point_observations);
  }

  Cost Accumulate(Vector<double, Dynamic>& gradient) {
    constexpr std::size_t kCostWorkItems = 128;
    const std::size_t camera_count = host_data_.cameras.size();
    const std::size_t point_count = host_data_.points.size();
    const std::size_t observation_count = host_data_.observations.size();
    const std::size_t camera_value_count = camera_count * 42;
    const std::size_t diagonal_value_count = camera_value_count + point_count * 12;
    const std::size_t cost_partial_count = CostPartialCount();

    hessian_.setZero();
    gradient.setZero();

    queue_.submit([&](sycl::handler& handler) {
      const auto cameras = camera_buffer_.template get_access<sycl::access::mode::read>(handler);
      const auto points = point_buffer_.template get_access<sycl::access::mode::read>(handler);
      const auto observations =
          observation_buffer_.template get_access<sycl::access::mode::read>(handler);
      auto evaluations = evaluation_buffer_.template get_access<sycl::access::mode::write>(handler);
      auto coupling = coupling_buffer_.template get_access<sycl::access::mode::write>(handler);
      auto costs = cost_buffer_.template get_access<sycl::access::mode::write>(handler);

      handler.parallel_for(sycl::range<1>(observation_count), [=](sycl::id<1> id) {
        const std::size_t index = id[0];
        const auto observation = observations[index];
        const auto camera = cameras[static_cast<std::size_t>(observation.camera)];
        const auto point = points[static_cast<std::size_t>(observation.point)];
        const double px = point.position[0];
        const double py = point.position[1];
        const double pz = point.position[2];
        double camera_point[3]{camera.translation[0], camera.translation[1],
                               camera.translation[2]};
        for (int col = 0; col < 3; ++col) {
          for (int row = 0; row < 3; ++row)
            camera_point[row] += camera.rotation[col * 3 + row] * point.position[col];
        }

        const double x = camera_point[0];
        const double y = camera_point[1];
        const double z_inverse = 1.0 / camera_point[2];
        const double z_inverse_squared = z_inverse * z_inverse;
        const double projection_x[3] = {bundle_adjustment::FocalX * z_inverse, 0.0,
                                        -bundle_adjustment::FocalX * x * z_inverse_squared};
        const double projection_y[3] = {0.0, bundle_adjustment::FocalY * z_inverse,
                                        -bundle_adjustment::FocalY * y * z_inverse_squared};
        const double residual[2] = {
            bundle_adjustment::FocalX * x * z_inverse + bundle_adjustment::PrincipalX -
                observation.measurement[0],
            bundle_adjustment::FocalY * y * z_inverse + bundle_adjustment::PrincipalY -
                observation.measurement[1]};

        sparse_detail::ObservationEvaluation evaluation;
        evaluation.cost = 0.5 * (residual[0] * residual[0] + residual[1] * residual[1]);
        evaluation.residual[0] = residual[0];
        evaluation.residual[1] = residual[1];
        for (int col = 0; col < 3; ++col) {
          const double r0 = camera.rotation[col * 3];
          const double r1 = camera.rotation[col * 3 + 1];
          const double r2 = camera.rotation[col * 3 + 2];
          const double point_jacobian_x =
              projection_x[0] * r0 + projection_x[1] * r1 + projection_x[2] * r2;
          const double point_jacobian_y =
              projection_y[0] * r0 + projection_y[1] * r1 + projection_y[2] * r2;
          evaluation.point_jacobian[col] = point_jacobian_x;
          evaluation.point_jacobian[3 + col] = point_jacobian_y;
          evaluation.camera_jacobian[3 + col] = point_jacobian_x;
          evaluation.camera_jacobian[9 + col] = point_jacobian_y;

          double rotation_derivative[3]{};
          for (int axis = 0; axis < 3; ++axis) {
            const double r_axis_0 = camera.rotation[axis];
            const double r_axis_1 = camera.rotation[3 + axis];
            const double r_axis_2 = camera.rotation[6 + axis];
            if (col == 0)
              rotation_derivative[axis] = -r_axis_1 * pz + r_axis_2 * py;
            else if (col == 1)
              rotation_derivative[axis] = r_axis_0 * pz - r_axis_2 * px;
            else
              rotation_derivative[axis] = -r_axis_0 * py + r_axis_1 * px;
          }
          const double rotation_jacobian_x =
              projection_x[0] * rotation_derivative[0] +
              projection_x[1] * rotation_derivative[1] +
              projection_x[2] * rotation_derivative[2];
          const double rotation_jacobian_y =
              projection_y[0] * rotation_derivative[0] +
              projection_y[1] * rotation_derivative[1] +
              projection_y[2] * rotation_derivative[2];
          evaluation.camera_jacobian[col] = rotation_jacobian_x;
          evaluation.camera_jacobian[6 + col] = rotation_jacobian_y;
        }
        evaluations[index] = evaluation;
        costs[index] = evaluation.cost;

        for (int col = 0; col < 3; ++col) {
          for (int row = 0; row < 6; ++row) {
            const double value =
                evaluation.camera_jacobian[row] * evaluation.point_jacobian[col] +
                evaluation.camera_jacobian[6 + row] * evaluation.point_jacobian[3 + col];
            coupling[static_cast<std::size_t>(observation.coupling_block) * 18 +
                     static_cast<std::size_t>(col * 6 + row)] = value;
          }
        }
      });
    });

    queue_.submit([&](sycl::handler& handler) {
      const auto evaluations =
          evaluation_buffer_.template get_access<sycl::access::mode::read>(handler);
      const auto camera_offsets =
          camera_offsets_buffer_.template get_access<sycl::access::mode::read>(handler);
      const auto camera_observations =
          camera_observations_buffer_.template get_access<sycl::access::mode::read>(handler);
      const auto point_offsets =
          point_offsets_buffer_.template get_access<sycl::access::mode::read>(handler);
      const auto point_observations =
          point_observations_buffer_.template get_access<sycl::access::mode::read>(handler);
      auto output = diagonal_buffer_.template get_access<sycl::access::mode::write>(handler);
      handler.parallel_for(sycl::range<1>(diagonal_value_count), [=](sycl::id<1> id) {
        const std::size_t output_index = id[0];
        double sum = 0.0;
        if (output_index < camera_value_count) {
          const std::size_t camera = output_index / 42;
          const std::size_t component = output_index % 42;
          for (std::size_t row = camera_offsets[camera]; row < camera_offsets[camera + 1];
               ++row) {
            const auto evaluation = evaluations[camera_observations[row]];
            if (component < 6) {
              const std::size_t axis = component;
              sum += evaluation.camera_jacobian[axis] * evaluation.residual[0] +
                     evaluation.camera_jacobian[6 + axis] * evaluation.residual[1];
            } else {
              const std::size_t matrix_index = component - 6;
              const std::size_t row_index = matrix_index / 6;
              const std::size_t col_index = matrix_index % 6;
              sum += evaluation.camera_jacobian[row_index] *
                         evaluation.camera_jacobian[col_index] +
                     evaluation.camera_jacobian[6 + row_index] *
                         evaluation.camera_jacobian[6 + col_index];
            }
          }
        } else {
          const std::size_t point_value = output_index - camera_value_count;
          const std::size_t point = point_value / 12;
          const std::size_t component = point_value % 12;
          for (std::size_t row = point_offsets[point]; row < point_offsets[point + 1]; ++row) {
            const std::size_t observation_index = point_observations[row];
            const auto evaluation = evaluations[observation_index];
            if (component < 3) {
              sum += evaluation.point_jacobian[component] * evaluation.residual[0] +
                     evaluation.point_jacobian[3 + component] * evaluation.residual[1];
            } else {
              const std::size_t matrix_index = component - 3;
              const std::size_t row_index = matrix_index / 3;
              const std::size_t col_index = matrix_index % 3;
              sum += evaluation.point_jacobian[row_index] *
                         evaluation.point_jacobian[col_index] +
                     evaluation.point_jacobian[3 + row_index] *
                         evaluation.point_jacobian[3 + col_index];
            }
          }
        }
        output[output_index] = sum;
      });
    });

    queue_.submit([&](sycl::handler& handler) {
      const auto costs = cost_buffer_.template get_access<sycl::access::mode::read>(handler);
      auto partials =
          cost_partials_buffer_.template get_access<sycl::access::mode::write>(handler);
      handler.parallel_for(sycl::range<1>(cost_partial_count), [=](sycl::id<1> id) {
        const std::size_t begin = id[0] * kCostWorkItems;
        const std::size_t end =
            begin + kCostWorkItems < observation_count ? begin + kCostWorkItems : observation_count;
        double sum = 0.0;
        for (std::size_t i = begin; i < end; ++i) sum += costs[i];
        partials[id] = sum;
      });
    });
    queue_.submit([&](sycl::handler& handler) {
      const auto partials =
          cost_partials_buffer_.template get_access<sycl::access::mode::read>(handler);
      auto total = total_cost_buffer_.template get_access<sycl::access::mode::write>(handler);
      handler.single_task([=] {
        double sum = 0.0;
        for (std::size_t i = 0; i < cost_partial_count; ++i) sum += partials[i];
        total[0] = sum;
      });
    });

    queue_.wait_and_throw();
    double cost = 0.0;
    {
      auto total = total_cost_buffer_.get_host_access();
      cost = total[0];
    }
    {
      auto diagonal = diagonal_buffer_.get_host_access();
      for (std::size_t camera = 0; camera < host_data_.cameras.size(); ++camera) {
        const std::size_t offset = camera * 42;
        for (int row = 0; row < 6; ++row) {
          gradient[static_cast<Index>(camera) * 6 + row] += diagonal[offset + row];
          for (int col = 0; col < 6; ++col)
            hessian_.template block<6, 6>(static_cast<Index>(camera),
                                          static_cast<Index>(camera))(row, col) =
                diagonal[offset + 6 + static_cast<std::size_t>(row * 6 + col)];
        }
      }
      const std::size_t point_offset = host_data_.cameras.size() * 42;
      const Index camera_gradient_offset = static_cast<Index>(host_data_.cameras.size() * 6);
      for (std::size_t point = 0; point < host_data_.points.size(); ++point) {
        const std::size_t offset = point_offset + point * 12;
        for (int row = 0; row < 3; ++row) {
          gradient[camera_gradient_offset + static_cast<Index>(point) * 3 + row] +=
              diagonal[offset + row];
          for (int col = 0; col < 3; ++col)
            hessian_.template block<3, 3>(static_cast<Index>(point),
                                          static_cast<Index>(point))(row, col) =
                diagonal[offset + 3 + static_cast<std::size_t>(row * 3 + col)];
        }
      }
    }
    {
      auto coupling = coupling_buffer_.get_host_access();
      auto& blocks = hessian_.template blocks<6, 3>();
      for (std::size_t block = 0; block < blocks.size(); ++block)
        for (int col = 0; col < 3; ++col)
          for (int row = 0; row < 6; ++row)
            blocks[block](row, col) =
                coupling[block * 18 + static_cast<std::size_t>(col * 6 + row)];
    }
    return Cost(cost, static_cast<int>(2 * host_data_.observations.size()));
  }

 private:
  std::size_t CostPartialCount() const {
    return (host_data_.observations.size() + 127) / 128;
  }

  sycl::queue& queue_;
  Hessian& hessian_;
  sparse_detail::HostData host_data_;
  sycl::buffer<sparse_detail::CameraData, 1> camera_buffer_;
  sycl::buffer<sparse_detail::PointData, 1> point_buffer_;
  sycl::buffer<sparse_detail::ObservationData, 1> observation_buffer_;
  sycl::buffer<std::size_t, 1> camera_offsets_buffer_;
  sycl::buffer<std::size_t, 1> camera_observations_buffer_;
  sycl::buffer<std::size_t, 1> point_offsets_buffer_;
  sycl::buffer<std::size_t, 1> point_observations_buffer_;
  sycl::buffer<sparse_detail::ObservationEvaluation, 1> evaluation_buffer_;
  sycl::buffer<double, 1> coupling_buffer_;
  sycl::buffer<double, 1> diagonal_buffer_;
  sycl::buffer<double, 1> cost_buffer_;
  sycl::buffer<double, 1> cost_partials_buffer_;
  sycl::buffer<double, 1> total_cost_buffer_;
};

}  // namespace tinyopt::benchmark::bundle_adjustment_sycl
