import React from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { usePlanContent } from '../Plan'
import {ByYearSchedule} from './ByYearSchedule/ByYearSchedule'
import {paramsInputValidateYearRange} from './Helpers/ParamInputValidate'

export const ParamsInputIncomeDuringRetirement = React.memo(() => {
  const content = usePlanContent()
  return (
    <div className="">
        <Contentful.RichText
          body={content.incomeDuringRetirement.intro.fields.body}
          p=""
        />
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
