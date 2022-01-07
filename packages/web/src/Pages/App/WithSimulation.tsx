import _ from 'lodash'
import React, {ReactNode} from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {
  useTPAWWorker,
  UseTPAWWorkerResult,
} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import {createContext} from '../../Utils/CreateContext'
import {StateObj} from '../../Utils/UseStateObj'
import {useTPAWParams} from './UseTPAWParams'

type _Value = {
  params:StateObj<TPAWParams>['value']
  setParams:StateObj<TPAWParams>['set']
  tpawResult: UseTPAWWorkerResult
  highlightPercentiles: number[]
}
const [Context, useSimulation] = createContext<_Value>('TPAW')

const numRuns = 500
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))


export {useSimulation}

export const WithSimulation = ({children}: {children: ReactNode}) => {
  const {params, setParams} = useTPAWParams()
  const {resultInfo: tpawResult} = useTPAWWorker(
    params,
    numRuns,
    percentiles
  )
  if (!tpawResult) return <></>
  const value = {params, setParams, tpawResult, highlightPercentiles}
  return <Context.Provider value={value}>{children}</Context.Provider>
}
