import { Params, getDefaultPlanParams } from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { ReactNode, useMemo } from 'react'
import { ParamsExtended, extendParams } from '../../TPAWSimulator/ExtentParams'
import {
  PlanParamsProcessed,
  processPlanParams,
} from '../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessed'
import {
  UseTPAWWorkerResult,
  useTPAWWorker,
} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import { createContext } from '../../Utils/CreateContext'
import { useAssertConst } from '../../Utils/UseAssertConst'
import {
  MarketData,
  synthesizeMarketDataForTesting,
} from '../Common/GetMarketData'
import {
  estimateCurrentPortfolioBalance,
  getHistoryForPortfolioBalanceEstimate,
} from './EstimateCurrentPortfolioBalance'
import { SetNonPlanParams, SetPlanParams, useParams } from './UseParams'
import { WithMarketData } from './WithMarketData'
import { WithURLPlanParams } from './WithURLPlanParams'
import { WithWASM, useWASM } from './WithWASM'

export type SimulationInfo = {
  currentTime: DateTime
  defaultParams: Params
  params: Params
  paramsProcessed: PlanParamsProcessed
  paramsExt: ParamsExtended
  currentPortfolioBalanceEstimate: ReturnType<
    typeof estimateCurrentPortfolioBalance
  >
  setPlanParams: SetPlanParams
  resetParams: (params: Params) => void
  setNonPlanParams: SetNonPlanParams
  setCurrentPortfolioBalance: (amount: number) => void
  tpawResult: UseTPAWWorkerResult
}

const [Context, useSimulation] = createContext<SimulationInfo>('Simulation')
export { useSimulation }

export const WithSimulation = React.memo(
  ({
    children,
    marketData,
  }: {
    children: ReactNode
    marketData: MarketData
  }) => {
    return (
      <WithURLPlanParams>
        <WithWASM>
          <_Body marketData={marketData}>{children}</_Body>
        </WithWASM>
      </WithURLPlanParams>
    )
  },
)

const _Body = ({
  children,
  marketData: marketDataIn,
}: {
  children: ReactNode
  marketData: MarketData
}) => {
  const paramInfo = useParams()
  const wasm = useWASM()

  const local = useMemo(() => {
    const { currentTime, params, setPlanParams: setPlanParamsIn } = paramInfo
    const marketData = params.nonPlan.dev.currentTimeFastForward
      .shouldFastForward
      ? synthesizeMarketDataForTesting(
          marketDataIn,
          currentTime.valueOf(),
          params.nonPlan.dev.currentTimeFastForward.marketDataExtensionStrategy,
          params.plan.advanced.annualReturns,
        )
      : marketDataIn
    const currentPortfolioBalanceEstimate = estimateCurrentPortfolioBalance(
      params,
      currentTime,
      marketData,
      wasm,
    )
    const paramsExt = extendParams(params, currentTime)
    const paramsProcessed = processPlanParams(
      paramsExt,
      currentPortfolioBalanceEstimate.value,
      marketData.latest,
    )

    const setPlanParams: SetPlanParams = (args) => {
      setPlanParamsIn((plan, params) => {
        const newPlan = typeof args === 'function' ? args(plan, params) : args
        if (newPlan === plan) return plan
        const { entered } = currentPortfolioBalanceEstimate
        newPlan.wealth.portfolioBalance = {
          isLastPlanChange: false,
          history: getHistoryForPortfolioBalanceEstimate(
            currentPortfolioBalanceEstimate,
            paramsExt.getDateTimeInCurrentTimezone,
          ),
          original: {
            amount: entered.amount,
            timestamp: entered.timestamp,
          },
        }
        return newPlan
      })
    }

    const setCurrentPortfolioBalance = (amount: number) =>
      setPlanParamsIn((plan) => {
        const clone = _.cloneDeep(plan)
        clone.wealth.portfolioBalance = {
          isLastPlanChange: true,
          amount,
          timestamp: currentTime.valueOf(),
        }
        return clone
      })

    return {
      paramsExt,
      paramsProcessed,
      currentPortfolioBalanceEstimate,
      marketData,
      setPlanParams,
      setCurrentPortfolioBalance,
    }
  }, [paramInfo, marketDataIn, wasm])
  useAssertConst([marketDataIn, wasm])

  const tpawResult = useTPAWWorker(local.paramsProcessed, local.paramsExt)

  const value = useMemo(() => {
    return {
      ...paramInfo,
      ...local,
      defaultParams: getDefaultPlanParams(paramInfo.currentTime),
      params: local.paramsProcessed.original,
      // Note, tpawResult will lag params. To get the exact params for the
      // result, use the params object inside tpawResult.
      tpawResult,
    }
  }, [paramInfo, local, tpawResult])
  if (!_hasValue(value)) return <></>
  return (
    <WithMarketData marketData={local.marketData}>
      <Context.Provider value={value}>{children}</Context.Provider>
    </WithMarketData>
  )
}

const _hasValue = (x: {
  tpawResult: UseTPAWWorkerResult | null
}): x is { tpawResult: UseTPAWWorkerResult } => x.tpawResult !== null
