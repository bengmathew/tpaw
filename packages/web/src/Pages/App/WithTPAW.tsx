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
  params: StateObj<TPAWParams>
  tpawResult: UseTPAWWorkerResult
  highlightPercentiles: number[]
}
const [Context, useTPAW] = createContext<_Value>('TPAW')

const numRuns = 500
const highlightPercentiles = [5, 25, 50, 75, 95]
const percentiles = _.sortBy(_.union(_.range(5, 95, 2), highlightPercentiles))


export {useTPAW}

export const WithTPAW = ({children}: {children: ReactNode}) => {
  const params = useTPAWParams()
  const {resultInfo: tpawResult} = useTPAWWorker(
    params.value,
    numRuns,
    percentiles
  )
  if (!tpawResult) return <></>
  const value = {params, tpawResult, highlightPercentiles}
  return <Context.Provider value={value}>{children}</Context.Provider>
}
