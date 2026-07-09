import { faCaretDown } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clix from 'clsx'
import React, { CSSProperties, useEffect, useRef, useState } from 'react'
import {
  PERCENTILE_PRESETS,
  SimulationPercentiles,
  useSimulationInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'

const _isSamePercentiles = (
  a: SimulationPercentiles,
  b: SimulationPercentiles,
) => a.low === b.low && a.mid === b.mid && a.high === b.high

const _label = (p: SimulationPercentiles) => (
  <span className="whitespace-nowrap">
    {p.low}
    <span className="lighten-2 px-px">/</span>
    {p.high}
  </span>
)

// Shows only the currently-selected percentile preset as a compact pill.
// Hovering (or tapping) reveals a popup listing all PERCENTILE_PRESETS to pick
// from. Changing the selection updates the simulation (via
// SimulationInfo.setPercentiles), which re-runs with the new bands.
export const PlanResultsChartCardPercentilePicker = React.memo(
  ({
    className,
    style,
    layout,
  }: {
    className?: string
    style?: CSSProperties
    layout: 'laptop' | 'desktop' | 'mobile'
  }) => {
    const planColors = usePlanColors()
    const { percentiles, setPercentiles } = useSimulationInfo()
    const [open, setOpen] = useState(false)
    const ref = useRef<HTMLDivElement>(null)

    // Close when tapping/clicking outside (hover-close doesn't fire on touch).
    useEffect(() => {
      if (!open) return
      const handler = (e: MouseEvent) => {
        if (ref.current && !ref.current.contains(e.target as Node))
          setOpen(false)
      }
      document.addEventListener('mousedown', handler)
      return () => document.removeEventListener('mousedown', handler)
    }, [open])

    const width = layout === 'mobile' ? 72 : 84
    const selected =
      PERCENTILE_PRESETS.find((x) => _isSamePercentiles(x, percentiles)) ??
      percentiles

    return (
      <div className={clix(className, 'flex justify-end')} style={style}>
        <div className="flex items-center gap-x-2">
          <div className="text-[13px] sm:text-[14px] lighten whitespace-nowrap">
            Percentiles
          </div>
          <div
            ref={ref}
            className="relative"
            onMouseEnter={() => setOpen(true)}
            onMouseLeave={() => setOpen(false)}
          >
            {/* Trigger: shows only the selected preset. */}
            <button
              className="flex items-center justify-center gap-x-1.5 rounded-full px-3 py-0.5 text-[13px] sm:text-[14px]"
              style={{
                width: `${width}px`,
                backgroundColor: planColors.shades.light[3].hex,
              }}
              onClick={() => setOpen((x) => !x)}
            >
              {_label(selected)}
              <FontAwesomeIcon
                className="text-[9px]"
                style={{
                  transitionProperty: 'transform',
                  transitionDuration: '200ms',
                  transform: open ? 'rotate(180deg)' : 'none',
                }}
                icon={faCaretDown}
              />
            </button>
            {/* Popup: all presets. pt-1.5 keeps the hover area continuous with
                the trigger so moving into the popup doesn't close it. */}
            <div
              className={clix(
                'absolute right-0 top-full z-50 pt-1.5',
                open ? 'opacity-100' : 'pointer-events-none opacity-0',
              )}
              style={{
                transitionProperty: 'opacity',
                transitionDuration: '150ms',
              }}
            >
              <div
                className="flex flex-col overflow-hidden rounded-xl shadow-lg"
                style={{ backgroundColor: planColors.shades.light[3].hex }}
              >
                {PERCENTILE_PRESETS.map((preset) => {
                  const isSel = _isSamePercentiles(preset, percentiles)
                  return (
                    <button
                      key={`${preset.low}-${preset.mid}-${preset.high}`}
                      className={clix(
                        'flex items-center justify-center px-4 py-1.5 text-[13px] sm:text-[14px]',
                        !isSel && 'hover:bg-gray-500/20',
                      )}
                      style={{
                        width: `${width}px`,
                        backgroundColor: isSel
                          ? planColors.shades.light[10].hex
                          : undefined,
                      }}
                      onClick={() => {
                        if (!isSel) setPercentiles(preset)
                        setOpen(false)
                      }}
                    >
                      {_label(preset)}
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  },
)
