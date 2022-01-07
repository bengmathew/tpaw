import _ from 'lodash'
import React, {useEffect, useState} from 'react'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput, useAmountInputState} from '../../Common/Inputs/AmountInput'

export const ParamsInputCurrentPortfolioValue = React.memo(() => {
  const {params} = useSimulation()
  const [key, setKey] = useState(0)
  useEffect(() => setKey(k => k + 1), [params])

  return (
    <div className="">
      <_Inner key={key} />
    </div>
  )
})

// This is needed for AmountInput to pull new values on resetDefault
const _Inner = React.memo(() => {
  const {params, setParams} = useSimulation()

  const valueState = useAmountInputState(params.savingsAtStartOfStartYear)
  return (
    <AmountInput
      className=""
      state={valueState}
      onAccept={amount => {
        if (amount === params.savingsAtStartOfStartYear) return
        const p = _.cloneDeep(params)
        p.savingsAtStartOfStartYear = amount
        setParams(p)
      }}
    />
  )
})
