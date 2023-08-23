// TODO: organize the return types better. THis became a mess to resolve a
// circular dependency issue that webpack was logging about: "Circular
// dependency between chunks with runtime...". But I don't think this was the
// root cause. That was resolved seperately, but I didn't back these out.

export namespace TPAWWorkerResultTypes {
  type SampleReturnsStatsForWindowSize = {
    n: number
    mean: number
    ofLog: {
      mean: number
      variance: number
      standardDeviation: number
      n: number
    }
  }

  export type GetSampledResultStats = {
    oneYear: SampleReturnsStatsForWindowSize
    fiveYear: SampleReturnsStatsForWindowSize
    tenYear: SampleReturnsStatsForWindowSize
    thirtyYear: SampleReturnsStatsForWindowSize
    perf: number
  }
}
