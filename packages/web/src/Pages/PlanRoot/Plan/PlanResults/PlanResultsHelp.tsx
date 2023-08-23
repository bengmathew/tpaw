import { faQuestion } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import Link from 'next/link'
import React, { useLayoutEffect, useState } from 'react'
import { useAssertConst } from '../../../../Utils/UseAssertConst'
import { useGetSectionURL } from '../Plan'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { setPlanInputBodyHeaderOnDoneSection } from '../PlanInput/PlanInputBody/PlanInputBodyHeader'
import { usePlanColors } from '../UsePlanColors'

export const PlanResultsHelp = React.memo(
  ({
    onHeight_const,
    targetDynamicSizing,
    duration,
    layout,
    section,
  }: {
    onHeight_const: (x: number) => void
    targetDynamicSizing: { inset: { bottom: number; left: number } }
    duration: number
    layout: 'laptop' | 'desktop' | 'mobile'
    section: PlanSectionName
  }) => {
    const getSectionURL = useGetSectionURL()

    const [element, setElement] = useState<HTMLAnchorElement | null>(null)
    useLayoutEffect(() => {
      if (!element) return
      const observer = new ResizeObserver((entries) =>
        // Not using entires because not available on iOS 14.
        onHeight_const(element.getBoundingClientRect().height),
      )
      observer.observe(element, { box: 'border-box' })
      return () => observer.disconnect()
    }, [element, onHeight_const])
    useAssertConst([onHeight_const])
    const planColors = usePlanColors()
    return (
      <>
        <Link
          ref={setElement}
          className={clsx('absolute flex items-center', planColors.results.fg)}
          style={{
            transitionProperty: 'opacity, transform, left, bottom',
            transitionDuration: `${duration}ms`,
            bottom: `${targetDynamicSizing.inset.bottom}px`,
            left: `${targetDynamicSizing.inset.left}px`,
            opacity: `${section === 'help' ? '0' : '1'}`,
            pointerEvents: section === 'help' ? 'none' : 'auto',
          }}
          onClick={() => {
            setPlanInputBodyHeaderOnDoneSection(section)
          }}
          href={getSectionURL('help')}
          shallow
        >
          <span
            className={clsx(
              `flex items-center justify-center rounded-full  mr-2`,
              layout === 'mobile'
                ? 'w-[20px] h-[20px] text-[12px]'
                : 'w-[25px] h-[25px] text-[18px]',
            )}
            style={{
              backgroundColor: planColors.results.darkBG,
              color: planColors.results.fgForDarkBG,
            }}
          >
            <FontAwesomeIcon
              className={` 
        ${layout === 'mobile' ? 'text-sm' : 'text-lg'}`}
              icon={faQuestion}
            />
          </span>
          <span className="font-semibold  text-base sm:text-lg1">
            Help me understand this
          </span>
        </Link>
      </>
    )
  },
)
