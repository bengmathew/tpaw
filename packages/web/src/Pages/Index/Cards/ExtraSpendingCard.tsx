import React from 'react'
import {checkYearRange, TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {StateObj} from '../../../Utils/UseStateObj'
import {ByYearSchedule} from '../ByYearSchedule/ByYearSchedule'
import {CardItem} from '../CardItem'

export const ExtraSpendingCard = React.memo(
  ({params: paramsObj}: {params: StateObj<TPAWParams>}) => {
    const {value: params} = paramsObj

    const numEntries =
      params.withdrawals.fundedByBonds.length +
      params.withdrawals.fundedByRiskPortfolio.length
    const subHeading = `${numEntries} ${numEntries === 1 ? 'entry' : 'entries'}`

    const warn = [
      ...params.withdrawals.fundedByBonds,
      ...params.withdrawals.fundedByRiskPortfolio,
    ].some(entry => checkYearRange(params, entry.yearRange) !== 'ok')

    const defaultYearRange = {
      start: 'retirement' as const,
      end: Math.min(params.age.end, params.age.retirement + 5),
    }

    return (
      <CardItem heading="Extra Spending" subHeading={subHeading} warn={warn}>
        <p className="mb-2">
          If you have extra spending needs during any years of your retirement,
          you can account for that here. For example, travel during early
          retirement, mortgage payments, and kids college tuition. You can enter
          a planned legacy as extra spending during the last year of retirement.
        </p>
        <p className="mb-2">
          You can categorize your extra spending as essential or discretionary.
          Essential spending will be funded with 100% bonds. Discretionary
          spending will be funded by your regular portfolio.
        </p>
        <ByYearSchedule
          className=""
          heading="Essential"
          addHeading="Add an Essential Expense"
          editHeading="Edit Essential Expense Entry"
          defaultYearRange={defaultYearRange}
          params={paramsObj}
          entries={params => params.withdrawals.fundedByBonds}
        />
        <ByYearSchedule
          className="mt-4"
          heading="Discretionary"
          addHeading="Add a Discretionary Expense"
          editHeading="Edit Discretionary Expense Entry"
          defaultYearRange={defaultYearRange}
          params={paramsObj}
          entries={params => params.withdrawals.fundedByRiskPortfolio}
        />
      </CardItem>
    )
  }
)
