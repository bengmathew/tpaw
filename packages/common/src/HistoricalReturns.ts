import _ from 'lodash'

const raw = [
  { year: 1871, stocks: 0.13905431, bonds: 0.035670483 },
  { year: 1872, stocks: 0.087811663, bonds: 0.015561673 },
  { year: 1873, stocks: 0.02034298, bonds: 0.114757246 },
  { year: 1874, stocks: 0.12500825, bonds: 0.167868952 },
  { year: 1875, stocks: 0.118238609, bonds: 0.156717926 },
  { year: 1876, stocks: -0.149164304, bonds: 0.048715308 },
  { year: 1877, stocks: 0.169730057, bonds: 0.2497398 },
  { year: 1878, stocks: 0.29615134, bonds: 0.174941611 },
  { year: 1879, stocks: 0.237991719, bonds: -0.122469893 },
  { year: 1880, stocks: 0.343651921, bonds: 0.13173245 },
  { year: 1881, stocks: -0.071933746, bonds: -0.033918711 },
  { year: 1882, stocks: 0.055871765, bonds: 0.055723236 },
  { year: 1883, stocks: 0.023113088, bonds: 0.123317381 },
  { year: 1884, stocks: -0.023055371, bonds: 0.165093382 },
  { year: 1885, stocks: 0.345350511, bonds: 0.085566342 },
  { year: 1886, stocks: 0.119399916, bonds: 0.022025203 },
  { year: 1887, stocks: -0.051459357, bonds: -0.022886134 },
  { year: 1888, stocks: 0.082091397, bonds: 0.105680177 },
  { year: 1889, stocks: 0.124177568, bonds: 0.089385937 },
  { year: 1890, stocks: -0.084472115, bonds: -0.006284823 },
  { year: 1891, stocks: 0.266030373, bonds: 0.105865312 },
  { year: 1892, stocks: -0.015188809, bonds: -0.049549462 },
  { year: 1893, stocks: -0.063908402, bonds: 0.201433581 },
  { year: 1894, stocks: 0.080576405, bonds: 0.103345999 },
  { year: 1895, stocks: 0.034592435, bonds: 0.009176227 },
  { year: 1896, stocks: 0.062552604, bonds: 0.084048523 },
  { year: 1897, stocks: 0.169339864, bonds: 0.008972849 },
  { year: 1898, stocks: 0.275202666, bonds: 0.040035116 },
  { year: 1899, stocks: -0.113094917, bonds: -0.121219722 },
  { year: 1900, stocks: 0.239383238, bonds: 0.061698511 },
  { year: 1901, stocks: 0.165856277, bonds: 0.000140906 },
  { year: 1902, stocks: -0.012315209, bonds: -0.067469381 },
  { year: 1903, stocks: -0.132751842, bonds: 0.072462283 },
  { year: 1904, stocks: 0.291338059, bonds: 0.004910918 },
  { year: 1905, stocks: 0.213054594, bonds: 0.039460539 },
  { year: 1906, stocks: -0.036322419, bonds: -0.02819569 },
  { year: 1907, stocks: -0.225169879, bonds: 0.043743077 },
  { year: 1908, stocks: 0.349742449, bonds: 0.014851759 },
  { year: 1909, stocks: 0.050158595, bonds: -0.07243636 },
  { year: 1910, stocks: 0.035937973, bonds: 0.108849708 },
  { year: 1911, stocks: 0.045985335, bonds: 0.04894225 },
  { year: 1912, stocks: -0.001044883, bonds: -0.06182729 },
  { year: 1913, stocks: -0.066429762, bonds: 0.04725387 },
  { year: 1914, stocks: -0.064058557, bonds: 0.025813065 },
  { year: 1915, stocks: 0.274338047, bonds: 0.02794096 },
  { year: 1916, stocks: -0.037917781, bonds: -0.087075486 },
  { year: 1917, stocks: -0.319210398, bonds: -0.150313463 },
  { year: 1918, stocks: 0.001762485, bonds: -0.107250048 },
  { year: 1919, stocks: 0.022727699, bonds: -0.136446996 },
  { year: 1920, stocks: -0.126189208, bonds: 0.058120394 },
  { year: 1921, stocks: 0.237614393, bonds: 0.25425432 },
  { year: 1922, stocks: 0.298966975, bonds: 0.045315199 },
  { year: 1923, stocks: 0.024094301, bonds: 0.037708143 },
  { year: 1924, stocks: 0.271164853, bonds: 0.057533791 },
  { year: 1925, stocks: 0.21653545, bonds: 0.018615772 },
  { year: 1926, stocks: 0.141147006, bonds: 0.089949952 },
  { year: 1927, stocks: 0.387451153, bonds: 0.046700985 },
  { year: 1928, stocks: 0.493477214, bonds: 0.023818508 },
  { year: 1929, stocks: -0.094269684, bonds: 0.062316808 },
  { year: 1930, stocks: -0.168951842, bonds: 0.106977499 },
  { year: 1931, stocks: -0.380278812, bonds: 0.119178749 },
  { year: 1932, stocks: 0.040196059, bonds: 0.184059272 },
  { year: 1933, stocks: 0.531039727, bonds: 0.025585512 },
  { year: 1934, stocks: -0.10710655, bonds: 0.028446701 },
  { year: 1935, stocks: 0.527106998, bonds: 0.025063804 },
  { year: 1936, stocks: 0.298931558, bonds: 0.002498737 },
  { year: 1937, stocks: -0.325606707, bonds: 0.030041385 },
  { year: 1938, stocks: 0.189489087, bonds: 0.058004215 },
  { year: 1939, stocks: 0.037984007, bonds: 0.044281711 },
  { year: 1940, stocks: -0.101714377, bonds: 0.030282826 },
  { year: 1941, stocks: -0.183369022, bonds: -0.12279712 },
  { year: 1942, stocks: 0.129713226, bonds: -0.048684514 },
  { year: 1943, stocks: 0.200691966, bonds: -0.005298894 },
  { year: 1944, stocks: 0.170022562, bonds: 0.011271917 },
  { year: 1945, stocks: 0.363003417, bonds: 0.016695187 },
  { year: 1946, stocks: -0.255345641, bonds: -0.139123032 },
  { year: 1947, stocks: -0.068927404, bonds: -0.086839394 },
  { year: 1948, stocks: 0.081985049, bonds: 0.022911089 },
  { year: 1949, stocks: 0.201119235, bonds: 0.044244155 },
  { year: 1950, stocks: 0.245138388, bonds: -0.072616041 },
  { year: 1951, stocks: 0.168262728, bonds: -0.025465871 },
  { year: 1952, stocks: 0.142581243, bonds: 0.010770555 },
  { year: 1953, stocks: 0.018769563, bonds: 0.048427859 },
  { year: 1954, stocks: 0.479252401, bonds: 0.020209459 },
  { year: 1955, stocks: 0.284589696, bonds: -0.000760972 },
  { year: 1956, stocks: 0.037794612, bonds: -0.044388706 },
  { year: 1957, stocks: -0.09094896, bonds: 0.031869181 },
  { year: 1958, stocks: 0.384396155, bonds: -0.056945231 },
  { year: 1959, stocks: 0.065453765, bonds: -0.023057241 },
  { year: 1960, stocks: 0.047577968, bonds: 0.099212854 },
  { year: 1961, stocks: 0.183027497, bonds: 0.012440915 },
  { year: 1962, stocks: -0.038619377, bonds: 0.047597618 },
  { year: 1963, stocks: 0.192707142, bonds: -0.004086925 },
  { year: 1964, stocks: 0.148861779, bonds: 0.0309713 },
  { year: 1965, stocks: 0.095166489, bonds: -0.009896302 },
  { year: 1966, stocks: -0.095315983, bonds: 0.01716788 },
  { year: 1967, stocks: 0.120357272, bonds: -0.057501399 },
  { year: 1968, stocks: 0.059684363, bonds: -0.025002896 },
  { year: 1969, stocks: -0.138693862, bonds: -0.111991903 },
  { year: 1970, stocks: 0.02136496, bonds: 0.139229001 },
  { year: 1971, stocks: 0.103888403, bonds: 0.051035271 },
  { year: 1972, stocks: 0.137211976, bonds: -0.011585366 },
  { year: 1973, stocks: -0.234589179, bonds: -0.058263121 },
  { year: 1974, stocks: -0.293907547, bonds: -0.069841875 },
  { year: 1975, stocks: 0.304365083, bonds: -0.002633017 },
  { year: 1976, stocks: 0.057316196, bonds: 0.06326751 },
  { year: 1977, stocks: -0.148171389, bonds: -0.04325264 },
  { year: 1978, stocks: 0.064012608, bonds: -0.077578773 },
  { year: 1979, stocks: 0.02862788, bonds: -0.133487064 },
  { year: 1980, stocks: 0.127335413, bonds: -0.099705237 },
  { year: 1981, stocks: -0.143773261, bonds: -0.052013844 },
  { year: 1982, stocks: 0.254772423, bonds: 0.38072524 },
  { year: 1983, stocks: 0.155427651, bonds: -0.003105255 },
  { year: 1984, stocks: 0.042575294, bonds: 0.110719734 },
  { year: 1985, stocks: 0.216777561, bonds: 0.223042901 },
  { year: 1986, stocks: 0.295221877, bonds: 0.224918018 },
  { year: 1987, stocks: -0.061736958, bonds: -0.064219429 },
  { year: 1988, stocks: 0.126998946, bonds: 0.014614149 },
  { year: 1989, stocks: 0.16927794, bonds: 0.095683788 },
  { year: 1990, stocks: -0.061325627, bonds: 0.038607007 },
  { year: 1991, stocks: 0.286116762, bonds: 0.133778296 },
  { year: 1992, stocks: 0.043425102, bonds: 0.07038691 },
  { year: 1993, stocks: 0.089591769, bonds: 0.100412149 },
  { year: 1994, stocks: -0.015970588, bonds: -0.09836676 },
  { year: 1995, stocks: 0.317357576, bonds: 0.210483105 },
  { year: 1996, stocks: 0.236169409, bonds: -0.034649731 },
  { year: 1997, stocks: 0.259378702, bonds: 0.132350779 },
  { year: 1998, stocks: 0.293540447, bonds: 0.103673861 },
  { year: 1999, stocks: 0.124976579, bonds: -0.111056101 },
  { year: 2000, stocks: -0.086240521, bonds: 0.143935586 },
  { year: 2001, stocks: -0.144505386, bonds: 0.048433436 },
  { year: 2002, stocks: -0.221476529, bonds: 0.103160274 },
  { year: 2003, stocks: 0.261541294, bonds: 0.011524272 },
  { year: 2004, stocks: 0.030013182, bonds: 0.00690476 },
  { year: 2005, stocks: 0.059203082, bonds: -0.012677039 },
  { year: 2006, stocks: 0.110891212, bonds: -8.92818e-6 },
  { year: 2007, stocks: -0.054698043, bonds: 0.089407593 },
  { year: 2008, stocks: -0.356495619, bonds: 0.147368371 },
  { year: 2009, stocks: 0.298481272, bonds: -0.09257338 },
  { year: 2010, stocks: 0.145187497, bonds: 0.043936627 },
  { year: 2011, stocks: 0.004690887, bonds: 0.128499318 },
  { year: 2012, stocks: 0.144014033, bonds: 0.007162894 },
  { year: 2013, stocks: 0.236561574, bonds: -0.073723951 },
  { year: 2014, stocks: 0.135804717, bonds: 0.118724101 },
  { year: 2015, stocks: -0.04749668, bonds: -0.011482725 },
  { year: 2016, stocks: 0.181588352, bonds: -0.036604118 },
  { year: 2017, stocks: 0.224585545, bonds: -0.010542083 },
  { year: 2018, stocks: -0.062028067, bonds: 0.001978359 },
  { year: 2019, stocks: 0.250408607, bonds: 0.083400178 },
  { year: 2020, stocks: 0.162378785, bonds: 0.05827981 },
]

const _logReturns = (returns: number[]) => returns.map((x) => Math.log(1 + x))

const _stats = (returns: number[]) => {
  const expectedValue = _.mean(returns)
  const variance =
    _.sumBy(returns, (x) => Math.pow(x - expectedValue, 2)) /
    (returns.length - 1)
  const standardDeviation = Math.sqrt(variance)

  return { returns, expectedValue, variance, standardDeviation }
}

// Empirically determined to be more accurate.
const deltaCalculated = { stocks: 0.0006162, bonds: 0.0000005 }

const statsFn = (returns: number[], delta: 'stocks' | 'bonds' | number = 0) => {
  const log = _stats(_logReturns(returns))

  const deltaNum = typeof delta === 'number' ? delta : deltaCalculated[delta]
  const convertExpectedToLog = (x: number) =>
    Math.log(1 + x) - log.variance / 2 + deltaNum

  const adjust = (targetExpectedValue: number) => {
    const targetLogExpectedValue = convertExpectedToLog(targetExpectedValue)
    const adjustmentLogExpected = log.expectedValue - targetLogExpectedValue
    const adjustedLogReturns = log.returns.map(
      (log) => log - adjustmentLogExpected,
    )
    return adjustedLogReturns.map((x) => Math.exp(x) - 1)
  }
  return {
    ..._stats(returns),
    log,
    convertExpectedToLog,
    adjust,
  }
}

export const historicalReturns = {
  raw,
  stocks: statsFn(
    raw.map((x) => x.stocks),
    'stocks',
  ),
  bonds: statsFn(
    raw.map((x) => x.bonds),
    'bonds',
  ),
  statsFn,
}
