import {TPAWSimulationForYear} from '../RunTPAWSimulation'
import {TPAWParams} from '../TPAWParams'

export type TPAWWorkerArgs =
  | {
      type: 'runSimulation'
      taskID: string
      args: {params: TPAWParams; numRuns: number}
    }
  | {type: 'sortRows'; taskID: string; args: {data: number[][]}}

export type TPAWWorkerResult =
  | {
      type: 'runSimulation'
      taskID: string
      result: {
        withdrawalsByYearsIntoRetirementByRun: number[][]
        firstYearOfSomeRun: TPAWSimulationForYear
        legacyByRun: number[]
      }
    }
  | {type: 'sortRows'; taskID: string; result: number[][]}

export type TPAWWorkerRunSimulationResult = Extract<
  TPAWWorkerResult,
  {type: 'runSimulation'}
>['result']
