import { Switch } from '@headlessui/react'
import clix from 'clsx'
import React from 'react'
import { gray } from '../../../Utils/ColorPalette'

export const SwitchAsToggle = React.memo(
  ({
    className = '',
    checked,
    setChecked,
    disabled,
    type = 'oneSided',
    style = {
      bg: {
        on: { color: gray['700'] },
        off: { color: gray['200'] },
      },
    },
    sizing = { width: 36, height: 20, gap: 3 },
  }: {
    className?: string
    checked: boolean
    setChecked: (value: boolean) => void
    disabled?: boolean
    style?: {
      bg: {
        on: { color: string }
        off: { color: string }
      }
    }
    sizing?: {
      width: number
      height: number
      gap: number
    }
    type?: 'twoSided' | 'oneSided'
  }) => {
    const dotS = sizing.height - sizing.gap * 2
    return (
      <Switch
        checked={checked}
        onChange={setChecked}
        disabled={disabled}
        className={clix(
          className,
          'relative inline-flex items-center rounded-full transition-colors',
        )}
        style={{
          width: `${sizing.width}px`,
          height: `${sizing.height}px`,
          backgroundColor:
            checked || type === 'twoSided'
              ? style.bg.on.color
              : style.bg.off.color,
        }}
      >
        <span
          className={clix(
            'inline-block transform bg-white rounded-full transition-transform',
          )}
          style={{
            transform: `translateX(${
              checked ? sizing.width - dotS - sizing.gap : sizing.gap
            }px)`,
            width: `${dotS}px`,
            height: `${dotS}px`,
          }}
        />
      </Switch>
    )
  },
)
