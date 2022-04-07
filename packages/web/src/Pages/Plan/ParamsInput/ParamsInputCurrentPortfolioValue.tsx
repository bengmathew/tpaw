import _ from 'lodash'
import React, { useEffect, useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput, useAmountInputState } from '../../Common/Inputs/AmountInput'
import { usePlanContent } from '../Plan'
import { ParamsInputBody, ParamsInputBodyProps } from './ParamsInputBody'

export const ParamsInputCurrentPortfolioValue = React.memo(
  (props: ParamsInputBodyProps) => {
    const {params} = useSimulation()
    const [key, setKey] = useState(0)
    useEffect(() => setKey(k => k + 1), [params])
    return <_Inner key={key} {...props} />
  }
)

// This is needed for AmountInput to pull new values on resetDefault
const _Inner = React.memo((props: ParamsInputBodyProps) => {
  const {params, setParams} = useSimulation()
  const content = usePlanContent()

  const valueState = useAmountInputState(params.savingsAtStartOfStartYear)
  return (
    <ParamsInputBody {...props}>
      <div className="">
        <Contentful.RichText
          body={content.currentPortfolioValue.intro.fields.body}
          p="p-base"
        />
        <AmountInput
          className="mt-4"
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
})
