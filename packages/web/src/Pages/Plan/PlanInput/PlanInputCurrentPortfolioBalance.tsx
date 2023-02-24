import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { paddingCSS } from '../../../Utils/Geometry'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from '../../Common/Inputs/AmountInput'
import { smartDeltaFnForAmountInput } from '../../Common/Inputs/SmartDeltaFnForAmountInput'
import { usePlanContent } from '../Plan'
import { planSectionLabel } from './Helpers/PlanSectionLabel'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputCurrentPortfolioBalance = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const { params, setParams } = useSimulation()
    const content = usePlanContent()
    const handleChange = (amount: number) => {
      amount = Math.max(amount, 0)
      if (amount === params.wealth.currentPortfolioBalance) return
      const clone = _.cloneDeep(params)
      clone.wealth.currentPortfolioBalance = amount
      setParams(clone)
    }
    return (
      <PlanInputBody {...props}>
        <div className="">
          <div
            className="params-card"
            style={{ padding: paddingCSS(props.sizing.cardPadding) }}
          >
            <Contentful.RichText
              body={
                content['current-portfolio-balance'].intro[
                  params.advanced.strategy
                ]
              }
              p="p-base"
            />
            <div className="mt-4 flex">
              <AmountInput
                className="text-input"
                prefix="$"
                value={params.wealth.currentPortfolioBalance}
                onChange={handleChange}
                decimals={0}
                modalLabel={planSectionLabel('current-portfolio-balance')}
              />
              <button
                className="ml-2 px-3"
                onClick={() =>
                  handleChange(
                    smartDeltaFnForAmountInput.increment(
                      params.wealth.currentPortfolioBalance,
                    ),
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
                      params.wealth.currentPortfolioBalance,
                    ),
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
  },
)
