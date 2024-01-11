import { DateTime } from 'luxon'
import React from 'react'
import { appPaths } from '../../../../AppPaths'
import { PlanPrintViewPageGroup } from './Helpers/PlanPrintViewPageGroup'
import { PlanPrintViewArgs } from './PlanPrintViewArgs'

export const PlanPrintViewFrontSection = React.memo(
  ({
    linkToEmbed,
    settings,
    planLabel,
  }: {
    linkToEmbed: URL | null
    settings: PlanPrintViewArgs['settings']
    planLabel: string | null
  }) => {
    return (
      <PlanPrintViewPageGroup className="relative" settings={settings}>
        <div className="">
          <div className="">
            <h2 className="font-bold text-[50px] leading-[50px]">
              {DateTime.now().toFormat('yyyy')}
            </h2>
            <h2 className="font-bold text-[25px] leading-[25px] ">
              {DateTime.now().toFormat('MMMM dd')}
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
            <h2 className="text-[18px] lighten-2 mt-3 ml-1">
              <a
                className=""
                target="_blank"
                href={linkToEmbed?.toString() ?? ''}
              >
                <span className="underline">Link to Plan</span>{' '}
              </a>
            </h2>
          </div>
          <div className="absolute bottom-[1in] print:bottom-0 right-[1in] flex flex-col items-end ">
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
