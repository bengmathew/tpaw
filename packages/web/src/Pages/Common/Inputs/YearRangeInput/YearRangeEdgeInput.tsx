import React, {useState} from 'react'
import {fGet} from '../../../../Utils/Utils'
import {YearRangeEdgeInputButton} from './YearRangeEdgeInputButton'
import {YearRangeEdgeInputPick} from './YearRangeEdgeInputPick'
import {YearRangeInputProps} from './YearRangeInput'

const duration = 1

export type YearRangeEdgeInputProps = React.ComponentProps<
  typeof YearRangeEdgeInput
>

export const YearRangeEdgeInput = React.memo(
  ({
    className = '',
    ...props
  }: {
    edge: 'start' | 'end'
  } & YearRangeInputProps) => {
    const [open, setOpen] = useState(false)

    const [referenceElement, setReferenceElement] =
      useState<HTMLDivElement | null>(null)

    return (
      <div className={`${className}`}>
        <YearRangeEdgeInputButton
          className=""
          onOpenPicker={() => setOpen(true)}
          setRef={setReferenceElement}
          {...props}
        />

        {open && (
          <YearRangeEdgeInputPick
            referenceElement={fGet(referenceElement)}
            onClose={() => setOpen(false)}
            {...props}
          />
        )}
      </div>
    )
  }
)
