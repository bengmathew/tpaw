import { assertFalse, fGet } from '@tpaw/common'
import React, { useMemo } from 'react'
import { useURLParam } from '../../Utils/UseURLParam'
import { PlanPrintView } from '../PlanRoot/PlanRootHelpers/PlanPrintView/PlanPrintView'
import { PlanPrintViewArgsServerSide } from '../PlanRoot/PlanRootHelpers/PlanPrintView/PlanPrintViewArgs'

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
    <PlanPrintView
      fixed={fixed}
      settings={settings}
      simulationResult={null}
      updateSettings={() => assertFalse()}
    />
  )
})

const testParams: PlanPrintViewArgsServerSide = {
  fixed: {
    planLabel: 'Test Plan',
    planParams: {
      v: 25,
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
          blockSizeForMonteCarloSampling: 60,
        },
        strategy: 'TPAW',
        annualInflation: {
          type: 'suggested',
        },
        historicalReturnsAdjustment: {
          bonds: {
            enableVolatility: true,
            adjustExpectedReturn: {
              type: 'toExpectedUsedForPlanning',
              correctForBlockSampling: true,
            },
          },
          stocks: {
            volatilityScale: 1.01,
            adjustExpectedReturn: {
              type: 'toExpectedUsedForPlanning',
              correctForBlockSampling: true,
            },
          },
        },
        expectedAnnualReturnForPlanning: {
          type: 'manual',
          bonds: 0.022,
          stocks: 0.03,
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
      CAPE: {
        oneOverCAPE: 0.03301945387553307,
        regressionAverage: 0.0536822832663609,
        suggested: 0.04184934611061025,
      },
      bondRates: {
        twentyYear: 0.0224,
      },
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
