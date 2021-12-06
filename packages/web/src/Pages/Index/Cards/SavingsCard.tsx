import React from 'react'
import {checkYearRange, TPAWParams} from '../../../TPAWSimulator/TPAWParams'
import {StateObj} from '../../../Utils/UseStateObj'
import {ByYearSchedule} from '../ByYearSchedule/ByYearSchedule'
import {CardItem} from '../CardItem'

export const SavingsCard = React.memo(
  ({params: paramsObj}: {params: StateObj<TPAWParams>}) => {
    const {value: params} = paramsObj

    const numEntries = params.savings.length
    const subHeading = `${numEntries} ${numEntries === 1 ? 'entry' : 'entries'}`

    const warn = params.savings.some(
      entry => checkYearRange(params, entry.yearRange) !== 'ok'
    )
    return (
      <CardItem
        heading="Future Savings and Retirement Income"
        subHeading={subHeading}
        warn={warn}
      >
        <p className="">
          Enter your planned future savings, and also income that will support
          you during retirement like pensions, Social Security, income from
          rental properties, etc.
        </p>
        <ByYearSchedule
          className=""
          heading={null}
          addHeading="Add to Savings"
          editHeading="Edit Savings Entry"
          defaultYearRange={{start: 'start', end: 'lastWorkingYear'}}
          params={paramsObj}
          entries={params => params.savings}
        />
      </CardItem>
    )
  }
)
