import React, {useState} from 'react'
import {Year} from '@tpaw/common'
import {SimpleRange} from '../../../../Utils/SimpleRange'
import {fGet} from '../../../../Utils/Utils'
import {YearInputButton} from './YearInputButton'
import {YearInputPick} from './YearInputPick'

export type YearInputProps = React.ComponentProps<typeof YearInput>

export type YearType =
  | 'now'
  | 'lastWorkingYear'
  | 'retirement'
  | 'maxAge'
  | 'numericAge'
  | 'forNumOfYears'

export const YearInput = React.memo(
  ({
    className = '',
    ...props
  }: {
    className?: string
    location: 'rangeStart' | 'rangeEnd' | 'standalone'
    value: Year | {numYears: number}
    onChange: (year: Year | {numYears: number}) => void
    toYFN: (value: Year | {numYears: number}) => number
    range: SimpleRange
    choices: YearType[]
    modalTextInputOnMobile:boolean
  }) => {
    const [open, setOpen] = useState(false)

    const [referenceElement, setReferenceElement] =
      useState<HTMLDivElement | null>(null)

    return (
      <div className={`${className}`}>
        <YearInputButton
          className=""
          onOpenPicker={() => setOpen(true)}
          setRef={setReferenceElement}
          {...props}
        />

        {open && (
          <YearInputPick
            referenceElement={fGet(referenceElement)}
            onClose={() => setOpen(false)}
            {...props}
          />
        )}
      </div>
    )
  }
)
