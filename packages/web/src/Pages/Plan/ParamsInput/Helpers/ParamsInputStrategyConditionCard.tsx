import {faCheck} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React from 'react'
import {TPAWParams} from '../../../../TPAWSimulator/TPAWParams'
import {paddingCSSStyle} from '../../../../Utils/Geometry'
import {noCase} from '../../../../Utils/Utils'
import {useSimulation} from '../../../App/WithSimulation'
import {ParamsInputBodyPassThruProps} from '../ParamsInputBody'

export const ParamsInputStrategyConditionCard = React.memo(
  ({
    className = '',
    tpaw,
    spaw,
    swr,
    children,
    props,
  }: {
    className?: string
    tpaw: boolean
    spaw: boolean
    swr: boolean
    children: React.ReactNode
    props: ParamsInputBodyPassThruProps
  }) => {
    const curr = useSimulation().params.strategy
    const active =
      (curr === 'TPAW' && tpaw) ||
      (curr === 'SPAW' && spaw) ||
      (curr === 'SWR' && swr)
      if(!active) return <></>
    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        {/* {!active && <h2
          className={`text-end text-sm w-full mb-2 font-medium${
            active ? '' : ''
          } `}
        >
          <span className="font-normal lighten">for</span>
          {tpaw && <_Strategy strategy="TPAW" />}
          {spaw && <_Strategy strategy="SPAW" />}
          {swr && <_Strategy strategy="SWR" />}
        </h2>} */}
        <div className={`${active ? '' : 'lighten-2 pointer-events-none'}`}>{children}</div>
      </div>
    )
  }
)

const _Strategy = React.memo(
  ({strategy}: {strategy: TPAWParams['strategy']}) => {
    const isCurr = strategy === useSimulation().params.strategy
    return (
      <span
        className={`ml-1.5 px-1 bg-gray-50  border-gray-200 border rounded-md inline-flex items-center
        ${isCurr ? 'font-bold' : ''}`}
      >
        {isCurr && (
          <FontAwesomeIcon
            className="text-theme1 mr-1 text-[11px]"
            icon={faCheck}
          />
        )}
        {strategyName(strategy)}
      </span>
    )
  }
)

export const strategyName = (strategy: TPAWParams['strategy']) => {
  switch (strategy) {
    case 'TPAW':
      return 'TPAW'
    case 'SPAW':
      return 'SPAW'
    case 'SWR':
      return 'SWR'
    default:
      noCase(strategy)
  }
}
