import { Month } from '@tpaw/common'
import React, { useState } from 'react'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { fGet } from '../../../../Utils/Utils'
import { MonthInputButton } from './MonthInputButton'
import { MonthInputPick } from './MonthInputPick'

export type MonthInputProps = React.ComponentProps<typeof MonthInput>

export type MonthType =
  | 'now'
  | 'lastWorkingMonth'
  | 'retirement'
  | 'maxAge'
  | 'numericAge'
  | 'calendarMonth'
  | 'forNumOfMonths'

export const MonthInput = React.memo(
  ({
    className = '',
    ...props
  }: {
    className?: string
    classNameForNumberInput?: string
    location: 'rangeStart' | 'rangeEnd' | 'standalone'
    valueClamped: Month | { numMonths: number }
    onChange: (month: Month | { numMonths: number }) => void
    toMFN: (value: Month | { numMonths: number }) => number
    range: SimpleRange
    choices: MonthType[]
    modalTextInputOnMobile: boolean
    getMonthLabel: (month: Month | { numMonths: number }) => string
  }) => {
    const [open, setOpen] = useState(false)

    const [referenceElement, setReferenceElement] =
      useState<HTMLDivElement | null>(null)

    return (
      <div className={`${className}`}>
        <MonthInputButton
          className=""
          onOpenPicker={() => setOpen(true)}
          setRef={setReferenceElement}
          {...props}
        />

        {open && (
          <MonthInputPick
            referenceElement={fGet(referenceElement)}
            onClose={() => setOpen(false)}
            {...props}
          />
        )}
      </div>
    )
  },
)
