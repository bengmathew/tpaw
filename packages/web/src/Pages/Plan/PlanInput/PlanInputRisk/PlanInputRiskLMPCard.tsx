import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { smartDeltaFn } from '../../../../Utils/SmartDeltaFn'
import { useSimulation } from '../../../App/WithSimulation'
import { AmountInput } from '../../../Common/Inputs/AmountInput'
import { Config } from '../../../Config'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'

// TODO:
// spendingCeiling should be >= lmp. Which should be enforced by the PlanParams guard.
// in the same way spendingFloor is.

export const PlanInputRiskLMPCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setParams } = useSimulation()
    const currLMP = params.risk.tpawAndSPAW.lmp

    const handleAmount = (amount: number) => {
      if (amount === currLMP) return
      const p = _.cloneDeep(params)
      p.risk.tpawAndSPAW.lmp = amount
      setParams(p)
    }

    if (Config.client.production) return <></>

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg">LMP</h2>
        <div className=""></div>
        <div className={`flex items-stretch gap-x-2 mt-4`}>
          <AmountInput
            className="text-input"
            prefix="$"
            value={currLMP}
            onChange={handleAmount}
            decimals={0}
            modalLabel={`LMP: TODO: rename?`}
          />
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(lmpDeltaFn.increment(currLMP))}
          >
            <FontAwesomeIcon className="text-base" icon={faPlus} />
          </button>
          <button
            className={`flex items-center px-2 `}
            onClick={() => handleAmount(lmpDeltaFn.decrement(currLMP))}
          >
            <FontAwesomeIcon className="text-base" icon={faMinus} />
          </button>
        </div>
      </div>
    )
  },
)

const lmpDeltaFn = smartDeltaFn([
  { value: 1000000, delta: 1000 },
  { value: 2000000, delta: 2500 },
])
