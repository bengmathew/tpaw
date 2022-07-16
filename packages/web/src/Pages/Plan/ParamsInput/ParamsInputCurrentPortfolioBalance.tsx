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
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputCurrentPortfolioBalance = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()
    const handleChange = (amount: number) => {
      amount = Math.max(amount, 0)
      if (amount === params.savingsAtStartOfStartYear) return
      const p = _.cloneDeep(params)
      p.savingsAtStartOfStartYear = amount
      setParams(p)
    }
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
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
              value={params.savingsAtStartOfStartYear}
              onChange={handleChange}
              decimals={0}
            />
            <button
              className="ml-2 px-3"
              onClick={() =>
                handleChange(
                  smartDeltaFnForAmountInput.increment(
                    params.savingsAtStartOfStartYear
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
                    params.savingsAtStartOfStartYear
                  )
                )
              }
            >
              <FontAwesomeIcon icon={faMinus} />
            </button>
          </div>
        </div>
      </ParamsInputBody>
    )
  }
)
