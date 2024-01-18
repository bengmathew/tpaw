import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { PlanParams } from '@tpaw/common'
import React from 'react'
import { Contentful } from '../../../../Utils/Contentful'
import { paddingCSS, paddingCSSStyleHorz } from '../../../../Utils/Geometry'
import { noCase } from '../../../../Utils/Utils'
import { usePlanContent } from '../../PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
import { PlanInputSummaryChoiceItem } from './Helpers/PlanInputSummaryChoiceItem'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputStrategy = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { planParams, updatePlanParams } = useSimulation()

    const content = usePlanContent()['strategy']
    const handleStrategy = (strategy: PlanParams['advanced']['strategy']) =>
      updatePlanParams('setStrategy', strategy)

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
              body={content.intro[planParams.advanced.strategy]}
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
    const { planParams } = useSimulation()
    const content = usePlanContent()['strategy']
    const label = (() => {
      switch (strategy) {
        case 'TPAW':
          return `TPAW — Total Portfolio Allocation and Withdrawal`
        case 'SPAW':
          return `SPAW — Savings Portfolio Allocation and Withdrawal`
        case 'SWR':
          return `SWR — Safe Withdrawal Rate`
        default:
          noCase(strategy)
      }
    })()
    const isModified = useIsCardModified(strategy)

    return (
      <div
        className={`${className} params-card outline-none relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <button
          className={`text-left  `}
          onClick={() => handleStrategy(strategy)}
        >
          <h2 className=" font-bold text-lg">
            <FontAwesomeIcon
              className="mr-2"
              icon={
                planParams.advanced.strategy === strategy
                  ? faCircleSelected
                  : faCircleRegular
              }
            />{' '}
            {label}
          </h2>
          <div className="mt-2">
            <Contentful.RichText
              body={content.cardIntro[strategy][planParams.advanced.strategy]}
              p="p-base"
            />
          </div>
        </button>
      </div>
    )
  },
)

export const useIsPlanInputStrategyModified = () => {
  const isTPAWCardModified = useIsCardModified('TPAW')
  const isSPAWCardModified = useIsCardModified('SPAW')
  const isSWRCardModified = useIsCardModified('SWR')
  return isSPAWCardModified || isSWRCardModified || isTPAWCardModified
}

const useIsCardModified = (type: PlanParams['advanced']['strategy']) => {
  const { planParams } = useSimulation()
  return type === 'TPAW' ? false : type === planParams.advanced.strategy
}

export const PlanInputStrategySummary = React.memo(
  ({ planParams }: { planParams: PlanParams }) => {
    return (
      <>
        {(['TPAW', 'SPAW', 'SWR'] as const).map((value) => (
          <PlanInputSummaryChoiceItem
            key={value}
            value={value}
            selected={(x) => planParams.advanced.strategy === x}
            label={(x) => x}
          />
        ))}
      </>
    )
  },
)
