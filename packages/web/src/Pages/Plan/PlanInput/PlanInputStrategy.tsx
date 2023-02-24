import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT, PlanParams } from '@tpaw/common'
import _ from 'lodash'
import React from 'react'
import { PlanParamsExt } from '../../../TPAWSimulator/PlanParamsExt'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS, paddingCSSStyleHorz } from '../../../Utils/Geometry'
import { assert, noCase } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { usePlanContent } from '../Plan'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputStrategy = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { params, setParams, paramsExt } = useSimulation()

    const content = usePlanContent()['strategy']
    const handleStrategy = (strategy: PlanParams['advanced']['strategy']) => {
      if (params.advanced.strategy === strategy) return
      setParams(cloneWithNewStrategy(paramsExt, strategy))
    }
    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className="px-2"
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, { scale: 0.5 }),
            }}
          >
            <Contentful.RichText
              body={content.intro[params.advanced.strategy]}
              p="p-base"
            />
          </div>

          <_StrategyCard
            className="mt-10"
            strategy="TPAW"
            handleStrategy={handleStrategy}
            props={props}
          />
          <_StrategyCard
            className="mt-10"
            strategy="SPAW"
            handleStrategy={handleStrategy}
            props={props}
          />
          <_StrategyCard
            className="mt-10"
            strategy="SWR"
            handleStrategy={handleStrategy}
            props={props}
          />
          <button
            className="mt-6 underline ml-2"
            onClick={() => handleStrategy('TPAW')}
          >
            Reset to Default
          </button>
        </div>
      </PlanInputBody>
    )
  },
)

const _StrategyCard = React.memo(
  ({
    className = '',
    props,
    strategy,
    handleStrategy,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
    strategy: PlanParams['advanced']['strategy']
    handleStrategy: (strategy: PlanParams['advanced']['strategy']) => void
  }) => {
    const { params } = useSimulation()
    const content = usePlanContent()['strategy']
    const label = (() => {
      switch (strategy) {
        case 'TPAW':
          return `TPAW – Total Portfolio Allocation and Withdrawal`
        case 'SPAW':
          return `SPAW – Savings Portfolio Allocation and Withdrawal`
        case 'SWR':
          return `SWR – Safe Withdrawal Rate`
        default:
          noCase(strategy)
      }
    })()

    return (
      <div
        className={`${className} params-card outline-none`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <button
          className={`text-left  `}
          onClick={() => handleStrategy(strategy)}
        >
          <h2 className=" font-bold text-lg">
            <FontAwesomeIcon
              className="mr-2"
              icon={
                params.advanced.strategy === strategy
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            {label}
          </h2>
          <div className="mt-2">
            <Contentful.RichText
              body={content.cardIntro[strategy][params.advanced.strategy]}
              p="p-base"
            />
          </div>
        </button>
      </div>
    )
  },
)

export function cloneWithNewStrategy(
  paramsExt: PlanParamsExt,
  strategy: PlanParams['advanced']['strategy'],
) {
  const { params } = paramsExt
  assert(params.advanced.strategy !== strategy)
  const clone = _.cloneDeep(params)
  clone.advanced.strategy = strategy
  if (
    strategy === 'SWR' &&
    clone.risk.swr.withdrawal.type === 'default'
  ) {
    clone.risk.swr.withdrawal = {
      type: 'asPercentPerYear',
      percentPerYear: DEFAULT_ANNUAL_SWR_WITHDRAWAL_PERCENT(
        paramsExt.numRetirementMonths,
      ),
    }
  }
  return clone
}
