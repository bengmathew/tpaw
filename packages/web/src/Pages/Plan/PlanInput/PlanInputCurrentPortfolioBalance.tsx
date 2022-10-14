import {faMinus, faPlus} from '@fortawesome/pro-regular-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput} from '../../Common/Inputs/AmountInput'
import {smartDeltaFnForAmountInput} from '../../Common/Inputs/SmartDeltaFnForAmountInput'
import {usePlanContent} from '../Plan'
import {planSectionLabel} from './Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputCurrentPortfolioBalance = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()
    const handleChange = (amount: number) => {
      amount = Math.max(amount, 0)
      if (amount === params.currentPortfolioBalance) return
      const p = _.cloneDeep(params)
      p.currentPortfolioBalance = amount
      setParams(p)
    }
    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className="params-card"
            style={{padding: paddingCSS(props.sizing.cardPadding)}}
          >
            <Contentful.RichText
              body={content['current-portfolio-balance'].intro[params.strategy]}
              p="p-base"
            />
            <div className="mt-4 flex">
              <AmountInput
                className="text-input"
                prefix="$"
                value={params.currentPortfolioBalance}
                onChange={handleChange}
                decimals={0}
                modalLabel={planSectionLabel('current-portfolio-balance')}
              />
              <button
                className="ml-2 px-3"
                onClick={() =>
                  handleChange(
                    smartDeltaFnForAmountInput.increment(
                      params.currentPortfolioBalance
                    )
                  )
                }
              >
                <FontAwesomeIcon icon={faPlus} />
              </button>
              <button
                className="px-3"
                onClick={() =>
                  handleChange(
                    smartDeltaFnForAmountInput.decrement(
                      params.currentPortfolioBalance
                    )
                  )
                }
              >
                <FontAwesomeIcon icon={faMinus} />
              </button>
            </div>
          </div>
        </div>
      </PlanInputBody>
    )
  }
)
