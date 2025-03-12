#include "./annualize_series.h"
#include "extern/doctest.h"
#include "src/utils/cuda_utils.h"


thrust::device_vector<FLOAT>
annualize_series(const thrust::device_vector<FLOAT> &series) {
  assert(series.size() >= 12);
  const auto window_iter =
      thrust::make_zip_iterator(thrust::make_tuple(series.begin(),
                                                   series.begin() + 1,
                                                   series.begin() + 2,
                                                   series.begin() + 3,
                                                   series.begin() + 4,
                                                   series.begin() + 5,
                                                   series.begin() + 6,
                                                   series.begin() + 7,
                                                   series.begin() + 8,
                                                   series.begin() + 9,
                                                   series.begin() + 10,
                                                   series.begin() + 11));
  thrust::device_vector<FLOAT> result(series.size() - 11);
  thrust::transform(window_iter,
                    window_iter + static_cast<long>(result.size()),
                    result.begin(),
                    [] __device__(const auto x) -> FLOAT {
                      return thrust::get<0>(x) + thrust::get<1>(x) +
                             thrust::get<2>(x) + thrust::get<3>(x) +
                             thrust::get<4>(x) + thrust::get<5>(x) +
                             thrust::get<6>(x) + thrust::get<7>(x) +
                             thrust::get<8>(x) + thrust::get<9>(x) +
                             thrust::get<10>(x) + thrust::get<11>(x);
                    });
  return result;
}

TEST_CASE("STAR::annualize_series") {
  const thrust::device_vector<FLOAT> series = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2};
  const std::vector<FLOAT> result =
      device_vector_to_host(annualize_series(series));
  CHECK(result.size() == 3);
  CHECK(result[0] == 78);
  CHECK(result[1] == 90);
  CHECK(result[2] == 90);
}
