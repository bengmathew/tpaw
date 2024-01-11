import { Listbox } from '@headlessui/react'
import { assert, fGet } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React, {
  ReactNode,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom'
import { Size, applyOriginToHTMLElement } from '../../../Utils/Geometry'
import { tailwindScreens } from '../../../Utils/TailwindScreens'
import { useSystemInfo } from '../../App/WithSystemInfo'

const duration = 300

const _ModalListbox = <T extends string | number>({
  className = '',
  choices,
  value,
  onChange,
  children: [buttonChild, renderGroup, renderChoice],
  isDisabled,
  align,
}: {
  className?: string
  value: T
  onChange: (value: T) => void
  children: [
    ReactNode,
    null | ((x: { choices: T[]; open: boolean }) => ReactNode),
    (x: {
      disabled: boolean
      active: boolean
      selected: boolean
      choice: T
    }) => ReactNode,
  ]
  choices: T[] | T[][]
  isDisabled: (choice: T) => boolean
  align: 'left' | 'right'
}) => {
  const referenceElementRef = useRef<HTMLButtonElement | null>(null)

  return (
    <Listbox value={value} onChange={(x) => onChange(x)}>
      {({ open }) => (
        <>
          <Listbox.Button className={className} ref={referenceElementRef}>
            {buttonChild}
          </Listbox.Button>
          <_Options
            value={value}
            choices={choices}
            referenceElement={referenceElementRef.current ?? null}
            align={align}
            open={open}
            isDisabled={isDisabled}
          >
            {[renderGroup, renderChoice]}
          </_Options>
        </>
      )}
    </Listbox>
  )
}

const __Options = <T extends string | number>({
  value,
  referenceElement,
  align,
  open,
  isDisabled,
  children: [renderGroup, renderChoice],
  choices: choicesIn,
}: {
  value: T
  open: boolean
  referenceElement: HTMLElement | null
  align: 'left' | 'right'
  children: [
    null | ((x: { choices: T[]; open: boolean }) => ReactNode),
    (x: {
      disabled: boolean
      active: boolean
      selected: boolean
      choice: T
    }) => ReactNode,
  ]
  isDisabled: (choice: T) => boolean
  choices: T[] | T[][]
}) => {
  let groups = [] as T[][]
  if (!_.isArray(choicesIn[0])) {
    assert(renderGroup === null)
    groups = [choicesIn as T[]]
  } else {
    groups = choicesIn as T[][]
  }

  const {windowSize} = useSystemInfo()

  const [size, setSize] = useState<Size | null>(null)
  const [show, setShow] = useState(false)

  const [openGroup, setOpenGroup] = useState(() =>
    groups.findIndex((x) => x.includes(value)),
  )
  const popperElementRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    setShow(open)
    if (open) {
      const { width, height } = fGet(size)
      const position = fGet(referenceElement).getBoundingClientRect()
      const origin = {
        y: Math.min(position.top, windowSize.height - height - 20),
        x:
          windowSize.width < tailwindScreens.sm
            ? 0
            : align === 'left'
            ? Math.min(position.left, windowSize.width - 10 - width)
            : Math.max(position.right - width, 10),
      }
      applyOriginToHTMLElement(origin, fGet(popperElementRef.current))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, size])

  useLayoutEffect(() => {
    const observer = new ResizeObserver(() => {
      setSize(fGet(popperElementRef.current).getBoundingClientRect())
    })
    observer.observe(fGet(popperElementRef.current))
    return () => observer.disconnect()
  }, [])

  const [opacity0AtTransitionEnd, setOpacity0AtTransitionEnd] = useState(true)
  const invisible = !show && opacity0AtTransitionEnd

  return (
    <>
      {ReactDOM.createPortal(
        <div
          className={clix(
            ' page fixed inset-0 z-50',
            // Not here, but in another setting, not doing this was causing an
            // issue on Safari where elements were not scrollable under this
            // even thought it was hidden.
            invisible && 'pointer-events-none',
          )}
          style={{
            visibility: invisible ? 'hidden' : 'visible',
            transitionProperty: 'opacity',
            transitionDuration: `${duration}ms`,
            opacity: show ? '1' : '0',
          }}
          onTransitionEnd={() => setOpacity0AtTransitionEnd(!show)}
        >
          <div className="fixed inset-0 bg-black opacity-70" />
          <div
            className={`flex absolute flex-col  rounded-xl   bg-planBG overflow-scroll max-h-[80vh] w-full sm:w-auto`}
            ref={popperElementRef}
            style={{
              transitionProperty: 'transform',
              transitionDuration: `${duration}ms`,
              transform: `translateY(${show ? '0' : '-10px'})`,
              boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)',
            }}
          >
            <Listbox.Options static>
              {groups.map((choices, i) => (
                <_Group
                  key={i}
                  open={openGroup === i}
                  setOpen={() => setOpenGroup(i)}
                  isDisabled={isDisabled}
                  choices={choices}
                >
                  {[renderGroup, renderChoice]}
                </_Group>
              ))}
            </Listbox.Options>
          </div>
        </div>,
        window.document.body,
      )}
    </>
  )
}

const __Group = <T extends string | number>({
  isDisabled,
  children: [renderGroup, renderChoice],
  choices,
  open,
  setOpen,
}: {
  children: [
    null | ((x: { choices: T[]; open: boolean }) => ReactNode),
    (x: {
      disabled: boolean
      active: boolean
      selected: boolean
      choice: T
    }) => ReactNode,
  ]
  isDisabled: (choice: T) => boolean
  choices: T[]
  open: boolean
  setOpen: () => void
}) => {
  return (
    <div className="">
      {renderGroup && (
        <button className="w-full" onClick={setOpen}>
          {renderGroup({ choices, open })}
        </button>
      )}
      {open &&
        choices.map((choice) => (
          <Listbox.Option
            className={'w-full'}
            key={choice}
            value={choice}
            disabled={isDisabled(choice)}
          >
            {({ active, selected, disabled }) => (
              <>{renderChoice({ active, selected, choice, disabled })}</>
            )}
          </Listbox.Option>
        ))}
    </div>
  )
}

const _Options = React.memo(__Options) as typeof __Options
const _Group = React.memo(__Group) as typeof __Group
export const ModalListbox = React.memo(_ModalListbox) as typeof _ModalListbox
