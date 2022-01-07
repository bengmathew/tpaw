import React from 'react'
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {paramsInputValidateYearRange} from './Helpers/ParamInputValidate'

export const ParamsInputRetirementIncome = React.memo(() => {
  return (
    <div className="">
      <p className="">
        Enter any income that will support you during retirement like pensions,
        Social Security, income from rental properties, etc.
      </p>
      <ByYearSchedule
        className=""
        type="afterRetirement"
        heading={null}
        addHeading="Add to Savings"
        editHeading="Edit Savings Entry"
        defaultYearRange={{start: 'start', end: 'lastWorkingYear'}}
        entries={params => params.retirementIncome}
        validateYearRange={(params, x) =>
          paramsInputValidateYearRange(params, 'retirementIncome', x)
        }
      />
    </div>
  )
})
