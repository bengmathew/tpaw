import _ from 'lodash'
import React, {useEffect, useState} from 'react'
import {formatCurrency} from '../../../Utils/FormatCurrency'
import {useTPAW} from '../../App/WithTPAW'
import {AmountInput, useAmountInputState} from '../../Common/Inputs/AmountInput'
import {CardItem} from '../CardItem'

export const CurrentPortfolioValueCard = React.memo(() => {
  const {value: params} = useTPAW().params
  const [key, setKey] = useState(0)
  useEffect(() => setKey(k => k + 1), [params])
  const subHeading = `${formatCurrency(params.savingsAtStartOfStartYear)}`

  return (
    <CardItem heading="Current Portfolio Value" subHeading={subHeading}>
      <_Inner key={key} />
    </CardItem>
  )
})

// This is needed for AmountInput to pull new values on resetDefault
const _Inner = React.memo(() => {
  const {value: params, set: setParams} = useTPAW().params
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
