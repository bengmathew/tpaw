import {faCircle as faCircleRegular} from '@fortawesome/pro-regular-svg-icons'
import {faCircle as faCircleSelected} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useEffect, useRef, useState} from 'react'
import {
  DEFAULT_SWR_WITHDRAWAL_PERCENT,
  resolveTPAWRiskPreset,
} from '../../../TPAWSimulator/DefaultParams'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {TPAWParamsExt} from '../../../TPAWSimulator/TPAWParamsExt'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS, paddingCSSStyleHorz} from '../../../Utils/Geometry'
import {useURLUpdater} from '../../../Utils/UseURLUpdater'
import {assert, fGet, noCase} from '../../../Utils/Utils'
import {useChartData} from '../../App/WithChartData'
import {useSimulation} from '../../App/WithSimulation'
import {ToggleSwitch} from '../../Common/Inputs/ToggleSwitch'
import {usePlanContent} from '../Plan'
import {useGetPlanChartURL} from '../PlanChart/UseGetPlanChartURL'
import {usePlanChartType} from '../PlanChart/UsePlanChartType'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputCompareStrategies = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {params, setParams, paramsExt} = useSimulation()

    const content = usePlanContent()['strategy']
    const handleStrategy = (strategy: TPAWParams['strategy']) => {
      if (params.strategy === strategy) return
      setParams(cloneWithNewStrategy(paramsExt, strategy))
    }
    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className="px-2"
            style={{
              ...paddingCSSStyleHorz(props.sizing.cardPadding, {scale: 0.5}),
            }}
          >
            <Contentful.RichText
              body={content.intro[params.strategy]}
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
          <_ComparisonCard className="mt-10 params-card " props={props} />
          <button
            className="mt-6 underline ml-2"
            onClick={() => handleStrategy('TPAW')}
          >
            Reset to Default
          </button>
        </div>
      </PlanInputBody>
    )
  }
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
    strategy: TPAWParams['strategy']
    handleStrategy: (strategy: TPAWParams['strategy']) => void
  }) => {
    const {params} = useSimulation()
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
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <button
          className={`text-left  `}
          onClick={() => handleStrategy(strategy)}
        >
          <h2 className=" font-bold text-lg">
            <FontAwesomeIcon
              className="mr-2"
              icon={
                params.strategy === strategy
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            {label}
          </h2>
          <div className="mt-2">
            <Contentful.RichText
              body={content.cardIntro[strategy][params.strategy]}
              p="p-base"
            />
          </div>
        </button>
      </div>
    )
  }
)

const _ComparisonCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {setCompareRewardRiskRatio, params} = useSimulation()
    const chartType = usePlanChartType()
    const urlUpdater = useURLUpdater()
    const getPlanChartURL = useGetPlanChartURL()

    const content = usePlanContent()['strategy']
    const chartData = useChartData().rewardRiskRatio

    // Remember the last non sharpe ratio chart type.
    const [lastNonShapeRatioChartType, setLastNonRewardRiskRatioChartType] =
      useState(chartType === 'reward-risk-ratio-comparison' ? null : chartType)

    useEffect(() => {
      if (chartType !== 'reward-risk-ratio-comparison')
        setLastNonRewardRiskRatioChartType(chartType)
    }, [chartType])

    // Flag for showing sharpe ratio.
    const [showRewardRiskRatio, setShowShapeRatio] = useState(false)

    // Sync byStrategy to flag.
    useEffect(() => {
      setCompareRewardRiskRatio(showRewardRiskRatio)
    }, [setCompareRewardRiskRatio, showRewardRiskRatio])

    // Sync chart type to flag and byStrategy.
    useEffect(() => {
      if (showRewardRiskRatio) {
        if (chartData) {
          urlUpdater.replace(getPlanChartURL('reward-risk-ratio-comparison'))
        }
      } else {
        if (
          chartType === 'reward-risk-ratio-comparison' &&
          lastNonShapeRatioChartType
        )
          urlUpdater.replace(getPlanChartURL(lastNonShapeRatioChartType))
      }
    }, [
      chartData,
      chartType,
      getPlanChartURL,
      lastNonShapeRatioChartType,
      showRewardRiskRatio,
      urlUpdater,
    ])

    // If chart type changes from outside.
    useEffect(() => {
      if (chartType !== 'reward-risk-ratio-comparison') {
        setShowShapeRatio(false)
      }
    }, [chartType])

    // Cleanup.
    const cleanupRef = useRef<() => void>()
    cleanupRef.current = () => {
      if (
        chartType === 'reward-risk-ratio-comparison' &&
        lastNonShapeRatioChartType
      ) {
        urlUpdater.replace(getPlanChartURL(lastNonShapeRatioChartType))
      }
      setCompareRewardRiskRatio(false)
    }
    useEffect(() => () => fGet(cleanupRef.current)(), [])

    return (
      <div
        className={`${className} params-card`}
        style={{padding: paddingCSS(props.sizing.cardPadding)}}
      >
        <h2 className="font-bold text-lg">Compare reward/risk ratio</h2>

        <div className="mt-2">
          <Contentful.RichText
            body={content.rewardRiskRatioIntro[params.strategy]}
            p="p-base"
          />
        </div>
        <div className="inline-flex gap-x-2 items-center py-1 mt-2">
          <button
            className={``}
            onClick={() => setShowShapeRatio(!showRewardRiskRatio)}
          >
            Show reward/risk ratio
          </button>
          <ToggleSwitch
            enabled={showRewardRiskRatio}
            setEnabled={setShowShapeRatio}
          />
        </div>
      </div>
    )
  }
)

export function cloneWithNewStrategy(
  paramsExt: TPAWParamsExt,
  strategy: TPAWParams['strategy']
) {
  const {params} = paramsExt
  assert(params.strategy !== strategy)
  const clone = _.cloneDeep(params)
  clone.strategy = strategy
  if (strategy === 'SWR' && clone.risk.swr.withdrawal.type === 'default') {
    clone.risk.swr.withdrawal = {
      type: 'asPercent',
      percent: DEFAULT_SWR_WITHDRAWAL_PERCENT(paramsExt.numRetirementYears),
    }
  }
  if (strategy !== 'TPAW' && clone.risk.useTPAWPreset) {
    clone.risk = resolveTPAWRiskPreset(clone.risk)
  }
  return clone
}
