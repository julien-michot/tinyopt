
echo "CPU Governor"
#sudo bash -c 'for ((i=0;i<$(nproc);i++)); do cpufreq-set -c $i -g performance; done'
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Building
echo "Building..."
make tinyopt_bench_ceres tinyopt_bench_dense -j5

# Benchmarks
echo "Running Ceres benchmarks"
./benchmarks/tinyopt_bench_ceres -r XML::out=../benchmarks/scripts/ceres.xml
echo "Running Tinyopt benchmarks"
./benchmarks/tinyopt_bench_tinyopt -r XML::out=../benchmarks/scripts/tinyopt.xml

echo "All Done"
#sudo bash -c 'for ((i=0;i<$(nproc);i++)); do cpufreq-set -c $i -g powersave; done'