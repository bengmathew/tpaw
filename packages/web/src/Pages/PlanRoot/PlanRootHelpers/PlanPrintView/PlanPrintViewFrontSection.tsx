import clsx from 'clsx'
import React from 'react'
import { appPaths } from '../../../../AppPaths'
import { CalendarDayFns } from '../../../../Utils/CalendarDayFns'
import { useSimulationResultInfo } from '../WithSimulation'
import { PlanPrintViewPageGroup } from './Helpers/PlanPrintViewPageGroup'
import { PlanPrintViewArgs } from './PlanPrintViewArgs'

export const PlanPrintViewFrontSection = React.memo(
  ({
    linkToEmbed,
    settings,
    planLabel,
  }: {
    linkToEmbed: { needsLink: boolean; link: string | null }
    settings: PlanPrintViewArgs['settings']
    planLabel: string | null
  }) => {
    const { datingInfo } = useSimulationResultInfo().simulationResult
      .planParamsNormOfResult

    return (
      <PlanPrintViewPageGroup className="relative" settings={settings}>
        <div className="">
          {/* Removed the element on !isDated was causing footer to move to next page.*/}
          <div className={clsx(!datingInfo.isDated && 'invisible')}>
            <h2 className="font-bold text-[50px] leading-[50px]">
              {datingInfo.isDated && `${datingInfo.nowAsCalendarDay.year}`}
            </h2>
            <h2 className="font-bold text-[25px] leading-[25px] ">
              {datingInfo.isDated &&
                CalendarDayFns.toStr(datingInfo.nowAsCalendarDay, {
                  skipYear: true,
                })}
            </h2>
          </div>
          <div className=" ">
            <h1 className="font-bold text-[65px] mt-[105px] leading-[65px]">
              Retirement <br /> Plan
            </h1>
            {planLabel && (
              <h2 className="font-semibold text-[40px] ml-[2px] mt-4">
                {planLabel}
              </h2>
            )}
            {!datingInfo.isDated && (
              <div className="text-[18px] mt-3 ml-1">
                <p className="font-font2">
                  This is a dateless plan. It is not tied to a calendar date.
                  Recommended for examples and not for personal planning.
                </p>
                <p className="font-font2 mt-3">
                  Created using market data as of{' '}
                  {CalendarDayFns.toStr(datingInfo.marketDataAsOfEndOfDayInNY)}
                </p>
              </div>
            )}
            {linkToEmbed.needsLink && (
              <h2 className="text-[18px] lighten-2 mt-3 ml-1">
                <a className="" target="_blank" href={linkToEmbed.link ?? ''}>
                  <span className="underline">Link to Plan</span>{' '}
                </a>
              </h2>
            )}
          </div>
          <div className="absolute bottom-[1in] print:bottom-0 right-[1in] flex flex-col items-end w-full">
            <h2 className="text-[25px] font-semibold ">TPAW Planner</h2>
            <a
              className="text-[18px] "
              target="_blank"
              href={appPaths.root().toString()}
            >
              tpawplanner.com
            </a>
          </div>
        </div>
      </PlanPrintViewPageGroup>
    )
  },
)
