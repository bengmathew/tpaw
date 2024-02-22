import { assertFalse, currentPlanParamsVersion, fGet } from '@tpaw/common'
import React, { useMemo } from 'react'
import { useURLParam } from '../../Utils/UseURLParam'
import { PlanPrintView } from '../PlanRoot/PlanRootHelpers/PlanPrintView/PlanPrintView'
import { PlanPrintViewArgsServerSide } from '../PlanRoot/PlanRootHelpers/PlanPrintView/PlanPrintViewArgs'
import { WithWASM } from '../PlanRoot/PlanRootHelpers/WithWASM'

export const ServerSidePrint = React.memo(() => {
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
      <PlanPrintView
        fixed={fixed}
        settings={settings}
        simulationResult={null}
        updateSettings={() => assertFalse()}
      />
    </WithWASM>
  )
})

const testParams: PlanPrintViewArgsServerSide = {
  fixed: {
    planLabel: 'Test Plan',
    planParams: {
      v: currentPlanParamsVersion,
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
                year: 2023,
                month: 11,
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
            monthOfBirth: {
              year: 1988,
              month: 11,
            },
          },
        },
        withPartner: false,
      },
      wealth: {
        futureSavings: {},
        portfolioBalance: {
          updatedHere: true,
          amount: 1547440.637135193,
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
          forMonteCarlo: {
            blockSize: 12 * 5,
            staggerRunStarts: true,
          },
        },
        strategy: 'TPAW',
        annualInflation: {
          type: 'suggested',
        },
        expectedReturnsForPlanning: {
          type: 'manual',
          bonds: 0.022,
          stocks: 0.03,
        },
        historicalMonthlyLogReturnsAdjustment: {
          standardDeviation: {
            stocks: { scale: 1.0 },
            bonds: { enableVolatility: true },
          },
          overrideToFixedForTesting: false,
        },
      },
      timestamp: 1704489409432,
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
    marketData: {
      inflation: {
        value: 0.0239,
      },
      sp500: {
        closingTime: 1706648400000,
        value: 4924.9702,
      },
      bondRates: {
        closingTime: 1706648400000,
        fiveYear: 0.0157,
        sevenYear: 0.0152,
        tenYear: 0.0149,
        twentyYear: 0.015,
        thirtyYear: 0.0154,
      },

      timestampMSForHistoricalReturns: Number.MAX_SAFE_INTEGER,
    },
    numOfSimulationForMonteCarloSampling: 500,
    ianaTimezoneName: 'America/Los_Angeles',
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
