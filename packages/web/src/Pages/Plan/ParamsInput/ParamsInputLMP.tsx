import {faMinus, faPlus} from '@fortawesome/pro-regular-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSSStyle} from '../../../Utils/Geometry'
import {smartDeltaFn} from '../../../Utils/SmartDeltaFn'
import {useSimulation} from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputLMP = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const content = usePlanContent()['lmp']

    return (
      <ParamsInputBody {...props} headingMarginLeft="reduced">
        <div className="">
          <_LMP className="" props={props} />
        </div>
      </ParamsInputBody>
    )
  }
)

const _LMP = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: ParamsInputBodyPassThruProps
  }) => {
    const {params, setParams} = useSimulation()

    const content = usePlanContent()['lmp']
    const handleAmount = (amount: number) => {
      if (amount === params.withdrawals.lmp) return
      const p = _.cloneDeep(params)
      p.withdrawals.lmp = amount
      setParams(p)
    }

    return (
      <div
        className={`${className} params-card`}
        style={{...paddingCSSStyle(props.sizing.cardPadding)}}
      >
        <div className="">
          <Contentful.RichText body={content.intro[params.strategy]} p="p-base" />
        </div>
        <div className={`flex items-stretch gap-x-2 mt-4`}>
          <AmountInput
            className="text-input"
            prefix="$"
            value={params.withdrawals.lmp}
            onChange={handleAmount}
            decimals={0}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(lmpDeltaFn.increment(params.withdrawals.lmp))
            }
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() =>
              handleAmount(lmpDeltaFn.decrement(params.withdrawals.lmp))
            }
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
        </div>
      </div>
    )
  }
)

const lmpDeltaFn = smartDeltaFn([
  {value: 1000000, delta: 1000},
  {value: 2000000, delta: 2500},
])
