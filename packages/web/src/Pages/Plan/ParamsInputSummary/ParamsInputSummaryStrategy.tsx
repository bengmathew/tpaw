import {faCaretDown} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useRef, useState} from 'react'
import {defaultSWRWithdrawalRate} from '../../../TPAWSimulator/DefaultParams'
import {TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {fGet} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {ContextMenu} from '../../Common/Modal/ContextMenu'
import {strategyName} from '../ParamsInput/Helpers/ParamsInputStrategyConditionCard'
import {ParamsInputType} from '../ParamsInput/Helpers/ParamsInputType'

export const ParamsInputSummaryStrategy = React.memo(
  ({
    className = '',
    setState,
  }: {
    className?: string
    setState: (state: ParamsInputType) => void
  }) => {
    const {params, setParams, paramsExt} = useSimulation()
    const [show, setShow] = useState(false)
    const ref = useRef<HTMLButtonElement>(null)
    const handleStrategy =
      (strategy: TPAWParams['strategy'], onHide: () => void) => () => {
        setParams(params => {
          if (params.strategy === strategy) return params
          const clone = _.cloneDeep(params)
          clone.strategy = strategy
          if (strategy === 'SWR' && clone.swrWithdrawal.type === 'default') {
            const percent = defaultSWRWithdrawalRate(
              paramsExt.numRetirementYears
            )
            clone.swrWithdrawal = {type: 'asPercent', percent}
          }
          return clone
        })
        onHide()
      }
    return (
      <button
        className={`${className} text-[18px] sm:text-lg font-bold py-1 flex items-center justify-end mb-2 w-full gap-x-1 `}
        onClick={() => setShow(true)}
        ref={ref}
      >
        <span className=" ">{strategyName(params.strategy)}</span>
        <FontAwesomeIcon className="" icon={faCaretDown} />
        {show && (
          <ContextMenu
            referenceElement={fGet(ref.current)}
            onClose={() => setShow(false)}
            align="right"
          >
            {onHide => (
              <div className="w-[100vw] sm:w-auto flex flex-col">
                <h2 className="font-bold  text-lg my-2 px-4 text-center">
                  Strategy
                </h2>
                <button
                  className={`text-left px-4 py-2 
                ${params.strategy === 'TPAW' ? 'bg-gray-200' : ''}`}
                  onClick={handleStrategy('TPAW', onHide)}
                >
                  <h2 className="font-bold">
                    TPAW <span className="text-sm font-normal">(default)</span>
                  </h2>
                  <h2 className="text-sm lighten">
                    Total Portfolio Allocation and Withdrawal
                  </h2>
                </button>
                <button
                  className={`text-left px-4 py-2 
                ${params.strategy === 'SPAW' ? 'bg-gray-200' : ''}`}
                  onClick={handleStrategy('SPAW', onHide)}
                >
                  <h2 className="font-bold">SPAW</h2>
                  <h2 className="text-sm lighten">
                    Savings Portfolio Allocation and Withdrawal
                  </h2>
                </button>
                <button
                  className={`text-left px-4 py-2 
                ${params.strategy === 'SWR' ? 'bg-gray-200' : ''}`}
                  onClick={handleStrategy('SWR', onHide)}
                >
                  <h2 className="font-bold">SWR</h2>
                  <h2 className="text-sm lighten">Safe Withdrawal Rate</h2>
                </button>
                <button
                  className="text-right px-5 py-3 text- font-bold "
                  onClick={() => {
                    onHide()
                    setState('compare-strategies')
                  }}
                >
                  Compare
                </button>
              </div>
            )}
          </ContextMenu>
        )}
      </button>
    )
  }
)
