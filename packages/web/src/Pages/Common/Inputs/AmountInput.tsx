import isMobile from 'is-mobile'
import React, {CSSProperties, useEffect, useRef, useState} from 'react'
import ReactDOM from 'react-dom'
import {NumericFormat} from 'react-number-format'
import {fGet} from '../../../Utils/Utils'

type Props = {
  className?: string
  value: number
  onChange: (value: number) => void
  disabled?: boolean
  prefix?: string
  suffix?: string
  decimals: number
  style?: CSSProperties
  onTransitionEnd?: () => void
}

export const AmountInput = React.memo(
  ({modalLabel, ...props}: Props & {modalLabel: string | null}) => {
    const [showModal, setShowModal] = useState(false)

    const [internalValue, setInternalValue] = useState(props.value)
    const ref = useRef<HTMLInputElement>(null)
    useEffect(() => setInternalValue(props.value), [props.value])

    return (
      <>
        <_AmountInput
          ref={ref}
          {...props}
          value={props.value}
          onFocus={() => {
            if (isMobile()) setShowModal(true)
          }}
          onEnter={() => {
            if(isMobile()) fGet(ref.current).blur()
          }}
        />

        {showModal &&
          modalLabel &&
          ReactDOM.createPortal(
            <div className="modal-base w-[100vw] h-[100vh] flex flex-col justify-center items-center">
              <div className="w-full h-full absolute bg-black opacity-60 z-0"></div>
              <div className="bg-pageBG z-10 w-[calc(100vw-50px)] py-5 px-5 rounded-lg ">
                <div className="w-">
                  <div className="">
                    <h2 className="text-lg font-bold mb-6">{modalLabel}</h2>
                    <_AmountInput
                      {...props}
                      value={internalValue}
                      onChange={setInternalValue}
                      onEnter={value => {
                        props.onChange(internalValue)
                        setShowModal(false)
                      }}
                      focusOnShow
                    />
                  </div>
                  <div className="mt-6 flex justify-end">
                    <button
                      className="btn-md "
                      onClick={() => {
                        setInternalValue(props.value)
                        setShowModal(false)
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      className="btn-md btn-dark"
                      onClick={() => {
                        props.onChange(internalValue)
                        setShowModal(false)
                      }}
                    >
                      Done
                    </button>
                  </div>
                </div>
              </div>
            </div>,
            window.document.body
          )}
      </>
    )
  }
)

type _Props = Props & {
  onEnter?: (value: number) => void
  onFocus?: () => void
  focusOnShow?: boolean
}
const _AmountInput = React.memo(
  React.forwardRef<HTMLInputElement, _Props>(
    (
      {
        className = '',
        style,
        value,
        onChange,
        disabled = false,
        prefix,
        suffix,
        decimals,
        onEnter,
        onFocus,
        focusOnShow = false,
        onTransitionEnd,
      }: _Props,
      forwardRef
    ) => {
      const [internalValue, setInternalValue] = useState<number | null>(value)
      useEffect(() => {
        setInternalValue(value)
      }, [value])
      const outputValue = internalValue === null ? 0 : internalValue

      return (
        <NumericFormat
          getInputRef={forwardRef}
          className={` ${className} `}
          style={style}
          onTransitionEnd={onTransitionEnd}
          autoFocus={focusOnShow}
          thousandSeparator={true}
          disabled={disabled}
          prefix={prefix}
          suffix={suffix}
          value={internalValue}
          decimalScale={decimals}
          fixedDecimalScale
          onValueChange={x => {
            setInternalValue(x.floatValue === undefined ? null : x.floatValue)
          }}
          onBlur={() => {
            // Don't rely on effect to change interval value. If inputValue is null
            // and outputValue is 0, and value is 0, inputValue won't update to 0.
            setInternalValue(outputValue)
            onChange(outputValue)
          }}
          onFocus={(e: React.FocusEvent<HTMLInputElement>) => {
            e.target.setSelectionRange(0, e.target.value.length)
            onFocus?.()
          }}
          onKeyDown={(e: React.KeyboardEvent) => {
            if (e.key === 'Enter') {
              // Don't rely on effect to change interval value. If inputValue is null
              // and outputValue is 0, and value is 0, inputValue won't update to 0.
              setInternalValue(outputValue)
              onChange(outputValue)
              onEnter?.(outputValue)
            }
          }}
        />
      )
    }
  )
)
