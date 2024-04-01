import React from 'react'
import { NormalizedLabeledAmountTimed } from '../../../../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { LabeledAmountTimedDisplay } from '../../../../Common/LabeledAmountTimedDisplay'

export const PlanInputSummaryLabeledAmountTimedList = React.memo(
  ({ entries }: { entries: NormalizedLabeledAmountTimed[] }) => {
    if (entries.length === 0) return <h2>None</h2>
    return (
      <ol className={`list-outside list-disc ml-3 grid gap-y-2 mt-1`}>
        {entries.map((entry, i) => (
          <li key={i} className="">
            <LabeledAmountTimedDisplay entry={entry} />
          </li>
        ))}
      </ol>
    )
  },
)
