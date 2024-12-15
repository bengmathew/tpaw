project('simulator-cuda', 'cuda', version:'1.0.0')

test_mode = get_option('test_mode')

sources = [
  'src/device_utils/get_random_index.cu',
  'src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_spaw_run_x_mfn_simulated_x_mfn.cu',
  'src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_swr_run_x_mfn_simulated_x_mfn.cu',
  'src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/cuda_process_tpaw_run_x_mfn_simulated_x_mfn.cu',
  'src/simulate/cuda_process_run_x_mfn_simulated_x_mfn/mertons_formula.cu',
  'src/simulate/run/run_common.cu',
  'src/simulate/run/run_result_padded.cu',
  'src/simulate/run/run_spaw.cu',
  'src/simulate/run/run_swr.cu',
  'src/simulate/run/run_tpaw.cu',
  'src/simulate/cuda_process_historical_returns.cu',
  'src/simulate/cuda_process_sampling.cu',
  'src/simulate/get_historical_returns_annualized_stats.cu',
  'src/simulate/get_result_cuda_not_array.cu',
  'src/simulate/pick_percentiles.cu',
  'src/simulate/plan_params_cuda_vectors.cu',
  'src/simulate/simulate.cu',
  'src/simulate/sort_run_result_padded.cu',
  'src/utils/annual_to_monthly_returns.cu',
  'src/utils/bench_utils.cu',
  'src/utils/cuda_test.cu',
  'src/utils/cuda_utils.cu',
  'src/utils/get_result_cuda_for_testing.cu',
  'src/utils/monthly_to_annual_returns.cu',
  'src/utils/print_public_types.cu',
  'src/simulator_cuda.cu',
  ]
if test_mode
  main_source = 'src/main_test.cu'
endif

public_headers = [
  'src/public_headers/historical_returns_cuda.h',
  'src/public_headers/numeric_types.h',
  'src/public_headers/opt_currency.h',
  'src/public_headers/plan_params_cuda.h',
  'src/public_headers/result_cuda.h',
  'src/public_headers/simulator_cuda.h',
  'src/public_headers/returns_stats.h',
  'src/public_headers/stocks_and_bonds_float.h',
  ]

gencode_args = [
  # '-gencode=arch=compute_80,code=sm_80',
  # A16 supports 8.6
  '-gencode=arch=compute_86,code=sm_86',
  # '-gencode=arch=compute_90,code=sm_90'
  ]

# std=c++20 was needed for templates in lambda closures.
# diag-suppress=445 is needed to suppress warnings about template parameters not being used in function signatures.
project_args = ['-rdc=true', '-Xcompiler=-Wextra,-Werror=missing-field-initializers', '--std=c++20', '-diag-suppress=445' ] + gencode_args
if not test_mode
  project_args += '-DDOCTEST_CONFIG_DISABLE'
endif
add_project_arguments(project_args, language : 'cuda')

link_args = gencode_args


if test_mode
  executable('simulator-cuda-test', sources + [main_source], link_args:link_args)
else 
  lib = library('simulator-cuda', sources, link_args:link_args, install:true)
  install_headers(public_headers, subdir:'simulator-cuda')
  # Full path to bindgen because meson in Dockerfile was not finding it otherwise, even though it's in the PATH.
  meson.add_install_script('/root/.cargo/bin/bindgen', '--raw-line', '#![allow(non_upper_case_globals)]', '--raw-line', '#![allow(non_camel_case_types)]', '--raw-line', '#![allow(non_snake_case)]', '/usr/local/include/simulator-cuda/simulator_cuda.h', '-o', '../../simulator-rust/src/lib/cuda_bridge.rs')
endif
 