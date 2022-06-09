import _ from 'lodash'
import React from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {paddingCSS} from '../../../Utils/Geometry'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput, useAmountInputState} from '../../Common/Inputs/AmountInput'
import {usePlanContent} from '../Plan'
import {ParamsInputBody, ParamsInputBodyPassThruProps} from './ParamsInputBody'

export const ParamsInputCurrentPortfolioBalance = React.memo(
  (props: ParamsInputBodyPassThruProps) => {
    const {params, setParams} = useSimulation()
    const content = usePlanContent()

    const valueState = useAmountInputState(params.savingsAtStartOfStartYear)
    return (
      <ParamsInputBody {...props} headingMarginLeft="normal">
        <div
          className="params-card"
          style={{padding: paddingCSS(props.sizing.cardPadding)}}
        >
          <Contentful.RichText
            body={content['current-portfolio-balance'].intro.fields.body}
            p="p-base"
          />
          <AmountInput
            className="mt-4"
            type="currency"
            state={valueState}
            onAccept={amount => {
              if (amount === params.savingsAtStartOfStartYear) return
              const p = _.cloneDeep(params)
              p.savingsAtStartOfStartYear = amount
              setParams(p)
            }}
          />
        </div>
      </ParamsInputBody>
    )
  }
)