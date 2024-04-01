import { faExclamationCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import React from 'react'
import { NormalizedDurationNotInThePast } from '../../../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeAmountAndTimingRecurring'
import { InMonthsInput } from './InMonthsInput'


export const MonthRangeDurationInput = React.memo(
  ({
    className, normValue, onChange,
  }: {
    className?: string
    normValue: NormalizedDurationNotInThePast
    onChange: (x: { inMonths: number} ) => void
  }) => {
    return (
      <div className={clsx(className, normValue.errorMsg && ' text-errorFG')}>
        <InMonthsInput
          className=""
          modalLabel={null}
          normValue={normValue}
          onChange={onChange} />
        {normValue.errorMsg && (
          <div className="mt-1">
            <FontAwesomeIcon className="" icon={faExclamationCircle} />{' '}
            {normValue.errorMsg}
          </div>
        )}
      </div>
    )
  }
)
