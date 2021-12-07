import {faArrowsV} from '@fortawesome/pro-regular-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React, {useEffect, useState} from 'react'
import Measure, {BoundingRect} from 'react-measure'
import {getDefaultParams} from '../../TPAWSimulator/DefaultParams'
import {fGet} from '../../Utils/Utils'
import {WithWindowWidth} from '../../Utils/WithWindowSize'
import {AppPage} from '../App/AppPage'
import {useTPAW} from '../App/WithTPAW'
import {AgeCard} from './Cards/AgeCard'
import {CurrentPortfolioValueCard} from './Cards/CurrentPortfolioValueCard'
import {ExtraSpendingCard} from './Cards/ExtraSpendingCard'
import {LegacyCard} from './Cards/LegacyCard'
import {ReturnsAndInflationCard} from './Cards/ReturnsAndInflationCard'
import {SavingsCard} from './Cards/SavingsCard'
import {SpendingCeilingValueCard} from './Cards/SpendingCeilingCard'
import {DistributionCanvasReact} from './Chart/DistributionCanvasReact'
import {LegacyDisplay} from './LegacyDisplay'
import {MainControls} from './MainControls'
import {Reset} from './Reset'
import {Share} from './Share'

const maxYScale = 1.1

export const Index = React.memo(() => {
  const {params, tpawResult, highlightPercentiles} = useTPAW()

  const [bounds, setBounds] = useState<BoundingRect | null>(null)

  const [maxY, setMaxY] = useState(0)
  useEffect(() => {
    if (maxY === 0 && tpawResult)
      setMaxY(Math.max(1, tpawResult.maxWithdrawal * maxYScale))
  }, [maxY, tpawResult])
  const handleRescale = () =>
    setMaxY(Math.max(1, tpawResult.maxWithdrawal * maxYScale))
  return (
    <WithWindowWidth>
      <AppPage title="TPAW Planner">
        <div className=" flex flex-col justify-start items-stretch">
          {/* <h2 className="font-bold text-xl ">ABOUT</h2> */}
          <p className=" ">
            Total Portfolio Allocation and Withdrawal (TPAW) is a retirement
            strategy that uses the total portfolio approach to calculate asset
            allocation and withdrawal.{' '}
            <Link href="/about">
              <a className="underline">Learn more</a>
            </Link>
            .
          </p>
          <h2 className="font-bold text-xl mt-6 ">
            SPENDING DURING RETIREMENT
          </h2>
          <p className="text-base lighten-2">
            Results from simulating your retirement {tpawResult.args.numRuns}{' '}
            times.
          </p>

          <Measure bounds onResize={({bounds}) => setBounds(fGet(bounds))}>
            {({measureRef}) => (
              <div
                className="h-[350px] sm:h-[450px] md:h-[475px] lg:h-[525px] -mx-1 mt-2 relative"
                ref={measureRef}
              >
                <div className=" absolute">
                  {bounds !== null && (
                    <DistributionCanvasReact
                      {...{tpawResult, maxY, highlightPercentiles}}
                      size={bounds}
                    />
                  )}
                </div>
                <LegacyDisplay className="" {...{params, tpawResult}} />
              </div>
            )}
          </Measure>

          <button
            className="self-end btn-sm btn-light "
            onClick={handleRescale}
          >
            <FontAwesomeIcon className="mr-2" icon={faArrowsV} />
            Rescale
          </button>
          <MainControls className="" params={params} />

          <h2 className="font-bold text-xl mt-10">INPUT</h2>
          <div className="self-start  gap-x-5 items-center mt-2">
            <Reset onReset={() => params.set(getDefaultParams())} />
            <Share params={params.value} />
          </div>
          <div className="flex flex-col items-start gap-y-5 mt-4">
            <AgeCard params={params} />
            <CurrentPortfolioValueCard />
            <SavingsCard params={params} />
            <ExtraSpendingCard params={params} />
            <SpendingCeilingValueCard {...{params, tpawResult}} />
            <LegacyCard />
            <ReturnsAndInflationCard params={params} />
          </div>

          <h2 className="font-bold text-xl mt-10">OUTPUT</h2>
          <div className="mt-2">
            <Link href="/tasks-for-this-year">
              <a className=" font-medium text-lg">
                View your tasks for this year{' '}
              </a>
            </Link>
          </div>
        </div>
      </AppPage>
    </WithWindowWidth>
  )
})
