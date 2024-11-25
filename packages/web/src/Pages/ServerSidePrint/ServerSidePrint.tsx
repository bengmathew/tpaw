import {
  CalendarMonthFns,
  MarketData,
  assertFalse,
  currentPlanParamsVersion,
  fGet,
  getZonedTimeFns,
  letIn,
} from '@tpaw/common'
import React, { useMemo } from 'react'
import { useURLParam } from '../../Utils/UseURLParam'
import { PlanPrintView } from '../PlanRoot/PlanRootHelpers/PlanPrintView/PlanPrintView'
import { PlanPrintViewArgsServerSide } from '../PlanRoot/PlanRootHelpers/PlanPrintView/PlanPrintViewArgs'
import { WithWASM } from '../PlanRoot/PlanRootHelpers/WithWASM'
import { WithMarketData } from '../PlanRoot/PlanRootHelpers/WithMarketData'

export const ServerSidePrint = React.memo(
  ({ marketData }: { marketData: MarketData.Data }) => {
    const urlParams = useURLParam('params')
    const { fixed, settings } = useMemo(
      () =>
        urlParams === 'test'
          ? testParams
          : (JSON.parse(fGet(urlParams)) as PlanPrintViewArgsServerSide),
      [urlParams],
    )

    return (
      <WithWASM>
        <WithMarketData marketData={marketData}>
          <PlanPrintView
            fixed={fixed}
            settings={settings}
            simulationResult={null}
            updateSettings={() => assertFalse()}
          />
        </WithMarketData>
      </WithWASM>
    )
  },
)

const testParams: PlanPrintViewArgsServerSide = {
  fixed: {
    planLabel: 'Test Plan',

    datingInfo: letIn(1704489409432, (nowAsTimestamp) => ({
      isDatedPlan: true,
      nowAsTimestamp,
      nowAsCalendarDay: getZonedTimeFns('America/Los_Angeles')(
        nowAsTimestamp,
      ),
    })),
    currentPortfolioBalanceAmount: 1547440,
    planParams: {
      v: currentPlanParamsVersion,
      timestamp: 1704489409432,
      datingInfo: { isDated: true },
      risk: {
        swr: {
          withdrawal: {
            type: 'default',
          },
        },
        spaw: {
          annualSpendingTilt: 0.008,
        },
        tpaw: {
          riskTolerance: {
            at20: 15,
            deltaAtMaxAge: -2,
            forLegacyAsDeltaFromAt20: 2,
          },
          timePreference: 0,
          additionalAnnualSpendingTilt: 0.004,
        },
        spawAndSWR: {
          allocation: {
            end: {
              stocks: 0.5,
            },
            start: {
              month: {
                type: 'now',
                monthOfEntry: {
                  isDatedPlan: true,
                  calendarMonth: { year: 2023, month: 11 },
                },
              },
              stocks: 0.5,
            },
            intermediate: {},
          },
        },
        tpawAndSPAW: {
          lmp: 0,
        },
      },
      people: {
        person1: {
          ages: {
            type: 'retiredWithNoRetirementDateSpecified',
            maxAge: {
              inMonths: 1200,
            },
            currentAgeInfo: {
              isDatedPlan: true,
              monthOfBirth: { year: 1988, month: 11 },
            },
          },
        },
        withPartner: false,
      },
      wealth: {
        futureSavings: {},
        portfolioBalance: {
          isDatedPlan: true,
          updatedHere: true,
          amount: 1547440,
        },
        incomeDuringRetirement: {},
      },
      results: {
        displayedAssetAllocation: {
          stocks: 0.1172,
        },
      },
      advanced: {
        sampling: {
          type: 'monteCarlo',
          data: {
            blockSize: { inMonths: 12 * 5 },
            staggerRunStarts: true,
          },
        },
        strategy: 'TPAW',
        annualInflation: {
          type: 'suggested',
        },
        returnsStatsForPlanning: {
          expectedValue: {
            empiricalAnnualNonLog: {
              type: 'fixed',
              bonds: 0.022,
              stocks: 0.03,
            },
          },
          standardDeviation: {
            stocks: { scale: { log: 1.0 } },
            bonds: { scale: { log: 0.0 } },
          },
        },
        historicalReturnsAdjustment: {
          standardDeviation: {
            bonds: { scale: { log: 1.0 } },
          },
          overrideToFixedForTesting: { type: 'none' },
        },
      },
      adjustmentsToSpending: {
        tpawAndSPAW: {
          legacy: {
            total: 0,
            external: {},
          },
          monthlySpendingFloor: null,
          monthlySpendingCeiling: null,
        },
        extraSpending: {
          essential: {},
          discretionary: {},
        },
      },
      dialogPositionNominal: 'done',
    },

    numOfSimulationForMonteCarloSampling: 500,

    randomSeed: 0,
  },
  settings: {
    isServerSidePrint: true,
    pageSize: 'Letter',
    linkToEmbed:
      'https://dev.tpawplanner.com/link?params=CuxGjzpy4VtcjT3vc49YUYcUcFrXXRbb',
    alwaysShowAllMonths: false,
  },
}
