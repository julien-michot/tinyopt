function(configure_tinyopt_features TARGET_NAME)

  # 0. Checks

  # 1. Definitions
  target_compile_definitions(${TARGET_NAME} INTERFACE
      HAS_EIGEN # mandatory
      $<$<NOT:$<BOOL:${TINYOPT_ENABLE_FORMATTERS}>>:TINYOPT_NO_FORMATTERS=1>
      $<$<BOOL:${TINYOPT_DISABLE_AUTODIFF}>:TINYOPT_NO_FORMATTERS=1>
      $<$<BOOL:${TINYOPT_DISABLE_NUMDIFF}>:TINYOPT_DISABLE_NUMDIFF=1>
      $<$<TARGET_EXISTS:fmt::fmt>:HAS_FMT>
      $<$<TARGET_EXISTS:Ceres::ceres>:HAS_CERES>
      $<$<TARGET_EXISTS:Sophus::Sophus>:HAS_SOPHUS>
      $<$<BOOL:${Sophus_FOUND}>:HAS_SOPHUS>
      $<$<BOOL:${LiePlusPlus_INCLUDE_DIR}>:HAS_LIEPLUSPLUS>
      $<$<TARGET_EXISTS:Catch2::Catch2WithMain>:CATCH2_MAJOR_VERSION=${CATCH2_MAJOR_VERSION}>
  )

  # 2. Include Directories
  target_include_directories(${TARGET_NAME} INTERFACE 
      $<$<BOOL:${Sophus_FOUND}>:${Sophus_INCLUDE_DIR}>
      $<$<BOOL:${LiePlusPlus_INCLUDE_DIR}>:${LiePlusPlus_INCLUDE_DIR}>
  )

  # # 3. Linking
  target_link_libraries(${TARGET_NAME} INTERFACE
      Eigen3::Eigen # mandatory
      $<$<TARGET_EXISTS:fmt::fmt>:fmt::fmt>
      $<$<TARGET_EXISTS:Ceres::ceres>:Ceres::ceres>
      $<$<TARGET_EXISTS:Sophus::Sophus>:Sophus::Sophus>
      $<$<TARGET_EXISTS:Catch2::Catch2WithMain>:Catch2::Catch2WithMain>
  )
endfunction()